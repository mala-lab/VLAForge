import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import random
import datetime
import yaml
from datetime import timedelta
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import inspect

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from model.VLAForge import VLAForge
from trainer.trainer import Trainer
from trainer.metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter

import numpy as np
from torch.utils.data import Sampler

CUDA_LAUNCH_BLOCKING=3

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--config_path', type=str,
                    default='/home/zhujiawen/VLAForge/config/train.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=False)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
parser.add_argument('--num_quires', type=int, default=128, help='')
args = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn_seed(worker_id):
    seed = 1024
    seed = seed + worker_id
    np.random.seed(seed)

class BalancedBatchSampler(Sampler):
    def __init__(self,
                 config,
                 dataset: DeepfakeAbstractBaseDataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.dataset = dataset

        self.label_list = self.dataset.label_list

        label_list = np.array(self.label_list)
        real_idx = np.argwhere(label_list == 0).flatten()
        fake_idx = np.argwhere(label_list == 1).flatten()

        self.real_generator = self.randomGenerator(real_idx)
        self.fake_generator = self.randomGenerator(fake_idx)

        self.batch_size = config["train_batchSize"]

        self.n_real = self.batch_size // 2
        self.n_fake = self.batch_size - self.n_real

        self.steps_per_epoch = 300

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):

        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_real):
                batch.append(next(self.real_generator))

            for _ in range(self.n_fake):
                batch.append(next(self.fake_generator))

            yield batch


def prepare_training_data(config):
    # Only use the blending dataset class in training
    if True:
        train_set = DeepfakeAbstractBaseDataset(config, mode='train')
        print(' Train set : FF++ 23')

    if config['ddp']:
        sample = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sample
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                train_set,
                worker_init_fn=worker_init_fn_seed,
                batch_sampler=BalancedBatchSampler(config, train_set),  # sampler=sampler,
                num_workers=int(config['workers']),
                collate_fn = train_set.collate_fn
            )

    return train_data_loader

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
        )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']

    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))

def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['local_rank'] = args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat

    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    setup_seed(1024)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model
    device = torch.cuda.current_device()

    model = VLAForge(clip_name=config['clip_model_name'],
               adapter_vit_name=config['vit_name'],
               num_quires=config['num_quires'],
               fusion_map=config['fusion_map'],
               mlp_dim=config['mlp_dim'],
               mlp_out_dim=config['mlp_out_fdim'],
               head_num=config['head_num'],
               device=device)

    count = count_parameters(model)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()

if __name__ == '__main__':
    main()
