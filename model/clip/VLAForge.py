import numpy as np
from sklearn import metrics
from torch import nn
from model.clip.clip import load
import torch
from .ForgePerceiver import ForgePerceiver
from .attn import RecAttnClip
from .layer import PostClipProcess, MaskPostXrayProcess
import torch.nn.functional as F
from trainer.metrics.base_metrics_class import calculate_metrics_for_train
from loss import FocalLoss, BinaryDiceLoss, BinaryFocalLoss
from einops import rearrange
from collections import OrderedDict
import open_clip
import math
import cv2
import os

def denormalize(img, mean, std):
    """反归一化 img: [3,H,W]"""
    mean = torch.tensor(mean).view(3,1,1).to(img.device)
    std = torch.tensor(std).view(3,1,1).to(img.device)
    return img * std + mean

class VLAForge(nn.Module):
    def __init__(self, clip_name,
                 adapter_vit_name,
                 num_quires,
                 fusion_map,
                 mlp_dim,
                 mlp_out_dim,
                 head_num,
                 device,
                 mode='video'):
        super().__init__()
        self.device = device
        self.clip_model, self.processor = load(clip_name, device=device, download_root='/data/cuixinjie/weights')
        self.adapter = ForgePerceiver(vit_name=adapter_vit_name, num_quires=num_quires, fusion_map=fusion_map, mlp_dim=mlp_dim,
                               mlp_out_dim=mlp_out_dim, head_num=head_num, device=self.device)
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, num_quires,device=self.device)  # 全部参数被冻结
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=num_quires).to(self.device)
        self.clip_post_process = PostClipProcess(num_quires=num_quires, embed_dim=768)

        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.trainable_layer = nn.Linear(1024, 768).to(device)
        self.conv = nn.Conv2d(in_channels=num_quires, out_channels=num_quires, kernel_size=1).to(device)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.classifier = nn.Linear(num_quires, 2, bias=False).to(device)

        self.num_quires = num_quires
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.mode = mode
        self._freeze()

    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_model' in name :
                param.requires_grad = False

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label'] #N
        xray = data_dict['xray']
        pred = pred_dict['cls']  #N2
        text_output = pred_dict['text_output']
        xray_pred = pred_dict['xray_pred']
        loss_intra = pred_dict['loss_intra']
        loss_clip = pred_dict['loss_clip']
        loss_orth =pred_dict['loss_orth']
        pred_map = pred_dict['pred_map']
        patch_label = data_dict['mask'].permute(0, 3, 1, 2)
        patch_label[patch_label > 0.5], patch_label[patch_label <= 0.5] = 1, 0
        criterion = nn.CrossEntropyLoss()
        loss_focal = FocalLoss()
        loss_dice = BinaryDiceLoss()
        loss1 = criterion(pred.float(), label) + criterion(text_output.float(), label)
        if xray is not None:
            loss_mse = F.mse_loss(xray_pred.squeeze().float(), patch_label.squeeze().float())

            loss_map = loss_dice(pred_map[:, 1, :, :], patch_label) + loss_focal(pred_map, patch_label)

            loss = 10 * loss1 + 200 * loss_mse + 20 * loss_intra + 10 * loss_clip  + 100 * loss_map + loss_orth

            loss_dict = {
                'cls': loss1,
                'xray': loss_mse,
                'intra': loss_intra,
                'loss_clip':loss_clip,
                'loss_orth':loss_orth,
                'overall': loss
            }
            return loss_dict
        else:
            loss_dict = {
                'cls': loss1,
                'overall': loss1
            }
            return loss_dict

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict, inference=False):
        images = data_dict['image']
        name = data_dict['name']

        clip_images = F.interpolate(
            images,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )

        clip_features = self.clip_model.extract_features(clip_images, self.adapter.fusion_map.values())
        img_features = self.clip_model.encode_image(clip_images)

        attn_biases, xray_preds, loss_adapter_intra = self.adapter(data_dict, clip_features, inference)

        map_feature = attn_biases[-1]
        B, H, Q, Hmap, Wmap = map_feature.shape
        f = map_feature.view(B, H, Q, -1)  # [B, H, Q, HW]
        f = F.normalize(f, dim=-1)  # 归一化
        loss_orth = 0
        for b in range(B):
            for h in range(H):
                sim = torch.matmul(f[b, h], f[b, h].T)  # [Q, Q] 相似度矩阵
                off_diag = sim - torch.diag(torch.diag(sim))
                loss_orth += off_diag.abs().mean()
        loss_orth = loss_orth / (B * H)

        patch_tokens, clip_output, loss_clip = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, True) # convey knowledge from adapter to clip (using adapter bias)

        hard_prompts_real = ["This is a real photo of id person."]
        hard_prompts_fake = ["This is a fake photo of id person."]

        text_feature = []
        for i in range(len(name)):
            id_emb = img_features[i].unsqueeze(0)
            id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)  # normalize embedding
            text_features_real = self.clip_model.encode_text(self.tokenizer(hard_prompts_real).to(self.device), id_emb)
            text_features_fake = self.clip_model.encode_text(self.tokenizer(hard_prompts_fake).to(self.device), id_emb)
            text_features_real = text_features_real / text_features_real.norm(dim=-1, keepdim=True)
            text_features_fake = text_features_fake / text_features_fake.norm(dim=-1, keepdim=True)
            text_feature.append(torch.stack([text_features_real, text_features_fake]))
        text_feature = torch.stack(text_feature)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature.squeeze(2)

        patch_tokens = self.trainable_layer(patch_tokens.float())
        anomaly_map = 100.0 * torch.matmul(patch_tokens, text_feature.float().transpose(1, 2))

        B, L, C = anomaly_map.shape
        H = int(np.sqrt(L))
        pred_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                 size=256, mode='bilinear', align_corners=True)
        pred_map = torch.softmax(pred_map, dim=1)
        anomaly_map = anomaly_map[:, :, 1].view(-1, 16, 16).unsqueeze(1)

        att_map = xray_preds[-1] * anomaly_map
        att_feat = self.conv(att_map)
        pooled = self.avg_pool(att_feat).view(-1, self.num_quires)
        text_output = self.classifier(pooled)
        text_output = torch.softmax(text_output, dim=1)

        xray_preds = [self.masked_xray_post_process(xray_pred) for xray_pred in xray_preds] # generate blending boundary mask

        clip_cls_output = self.clip_post_process(clip_output.float()).squeeze()   # N2
        clip_cls_output = torch.softmax(clip_cls_output, dim=1)

        outputs = {
            'xray_pred': xray_preds[-1],  # N 1 224 224
            'clip_cls_output': clip_cls_output,  # N 2
            'pred_map': pred_map,
            'text_output': text_output,
        }

        prob = 0.5 * outputs['clip_cls_output'][:, 1] + 0.5 * outputs['text_output'][:, 1]
        pred_dict = {
            'cls': outputs['clip_cls_output'],
            'prob': prob,
            'xray_pred': outputs['xray_pred'],
            'loss_intra': loss_adapter_intra,
            'loss_clip':loss_clip,
            'loss_orth': loss_orth,
            'pred_map': pred_map,
            'text_output': text_output,
        }

        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # acc
            _, prediction_class = torch.max(outputs['clip_cls_output'], 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

        return pred_dict
