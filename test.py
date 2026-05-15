"""
eval pretained model.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import random
import yaml
from tqdm import tqdm
from trainer.metrics.utils import get_test_metrics
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from model.VLAForge import VLAForge

import argparse

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhujiawen/VLAForge/config/test.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='/home/zhujiawen/DFD/logs/ds_ _2025-09-25-11-34-16/test/avg/ckpt_best.pth')
args = parser.parse_args()

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


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
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

def test_one_dataset(model, data_loader):
    prediction_lists = []
    # feature_lists = []
    label_lists = []

    # pred_map_lists = []
    # img_list = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to("cuda"), label.to("cuda")
        if mask is not None:
            data_dict['mask'] = mask.to("cuda")
        if landmark is not None:
            data_dict['landmark'] = landmark.to("cuda")

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)

        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())

    
    return np.array(prediction_lists), np.array(label_lists)  #,np.array(feature_lists)

def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps = test_one_dataset(model, test_data_loaders[key])
        print(f'name {data_dict.keys()}')
        # compute metric for each dataset
        print("---------------------------")
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):  #epoch,
    predictions = model(data_dict, inference=True)  #epoch,
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)

    device = torch.cuda.current_device()
    print(device)

    model = VLAForge(clip_name=config['clip_model_name'],
               adapter_vit_name=config['vit_name'],
               num_quires=config['num_quires'],
               fusion_map=config['fusion_map'],
               mlp_dim=config['mlp_dim'],
               mlp_out_dim=config['mlp_out_dim'],
               head_num=config['head_num'],
               device=device)

    try:
        epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
    except:
        epoch = 0
        ckpt = torch.load(weights_path, map_location="cuda")
        model.load_state_dict(ckpt, strict=True)
        model.to("cuda")

        print(f'===> Load {weights_path} done!')

        # start testing
        best_metric = test_epoch(model, test_data_loaders)
        print('===> Test Done!')

if __name__ == '__main__':
    main()
