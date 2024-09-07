import cv2
import os
import torch
import os.path as osp
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import mscoco.dataloader
from misc import pyutils, torchutils
import os

import math
import numpy as np

def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')(n_classes=80)
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()
    model.cuda()


    train_dataset = mscoco.dataloader.COCOClassificationDataset(
        image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        labels_path='./mscoco/single_positive_cls_labels.npy', 
        resize_long=(320, 640), hor_flip=True, crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=1,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True)
    

    if args.loss_type == 'llcp':
        llcp_label = torch.from_numpy(np.load(os.path.join(args.work_space, 'corrected_train_labels_spl.npy')))

    activated_class_labels = np.zeros_like(train_dataset.labels)
    
    with torch.no_grad():
        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)
            idx = pack['idx']
            x = model(img)
            preds = torch.sigmoid(x)

            if 'plt' in args.activation_type:
                label = llcp_label[idx].cuda()
            if 'pst' in args.activation_type:
                activated_class_labels[idx] = (preds + label >= args.pred_th).float().cpu()
            else:
                activated_class_labels[idx] = label.cpu()

    print(np.sum(activated_class_labels))
    np.save(os.path.join(args.work_space, 'activated_class_labels.npy'), activated_class_labels)