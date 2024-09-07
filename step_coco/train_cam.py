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

# for EM
LOG_EPSILON = 1e-5

def inverse_sigmoid(p):
    p = np.minimum(p, 1 - LOG_EPSILON)
    p = np.maximum(p, LOG_EPSILON)
    return np.log(p / (1-p))
def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg

class LabelEstimator(torch.nn.Module):
    
    def __init__(self, args, observed_label_matrix, estimated_labels):
        
        super(LabelEstimator, self).__init__()
        print('initializing label estimator')
        
        # Note: observed_label_matrix is assumed to have values in {-1, 0, 1} indicating 
        # observed negative, unknown, and observed positive labels, resp.
        
        num_examples = int(np.shape(observed_label_matrix)[0])
        observed_label_matrix = np.array(observed_label_matrix).astype(np.int8)
        total_pos = np.sum(observed_label_matrix == 1)
        total_neg = np.sum(observed_label_matrix == 0)
        print('observed positives: {} total, {:.1f} per example on average'.format(total_pos, total_pos / num_examples))
        print('observed negatives: {} total, {:.1f} per example on average'.format(total_neg, total_neg / num_examples))
        
        if estimated_labels is None:
            # initialize unobserved labels:
            w = 0.1
            q = inverse_sigmoid(0.5 + w)
            param_mtx = q * (2 * torch.rand(num_examples, args.num_classes) - 1)
            
            # initialize observed positive labels:
            init_logit_pos = inverse_sigmoid(0.995)
            idx_pos = torch.from_numpy((observed_label_matrix == 1).astype(np.bool))
            param_mtx[idx_pos] = init_logit_pos
            
            # initialize observed negative labels:
            init_logit_neg = inverse_sigmoid(0.005)
            idx_neg = torch.from_numpy((observed_label_matrix == -1).astype(np.bool))
            param_mtx[idx_neg] = init_logit_neg
        else:
            param_mtx = inverse_sigmoid(torch.FloatTensor(estimated_labels))
        
        self.logits = torch.nn.Parameter(param_mtx)
        
    def get_estimated_labels(self):
        with torch.set_grad_enabled(False):
            estimated_labels = torch.sigmoid(self.logits)
        estimated_labels = estimated_labels.clone().detach().cpu().numpy()
        return estimated_labels
    
    def forward(self, indices):
        x = self.logits[indices, :]
        x = torch.sigmoid(x)
        return x

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].float().cuda(non_blocking=True)

            x = model(img)
            loss = F.binary_cross_entropy_with_logits(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')(n_classes=80)


    train_dataset = mscoco.dataloader.COCOClassificationDataset(
        image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        labels_path='./mscoco/single_positive_cls_labels.npy', 
        resize_long=(320, 640), hor_flip=True, crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    # val_dataset = mscoco.dataloader.COCOClassificationDataset(
    #     image_dir = osp.join(args.mscoco_root,'val2014/'),
    #     anno_path= osp.join(args.mscoco_root,'annotations/instances_val2014.json'),
    #     labels_path='./mscoco/val_labels.npy',crop_size=512)
    # val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if args.loss_type == 'role':
        model_g = LabelEstimator(args, train_dataset.labels, None)
        model_g = torch.nn.DataParallel(model_g).cuda()
        model_g.train()
        expected_num_pos = 2.9

    param_groups = model.trainable_parameters()

    # if args.loss_type == 'role':
    #     optimizer = torchutils.PolyOptimizer([
    #         {'params': param_groups[0] + param_groups[1], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    #         {'params': model_g.parameters(), 'lr': args.lr_mult*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    #     ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    
    if args.loss_type == 'role':
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
            {'params': list(model_g.parameters()) + param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    
    else:
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
            {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    llcp_count = 0

    if args.loss_type == 'emapl':
        unlabel_num = []
        for i in range(train_dataset.labels.shape[1]):
            unlabel_num.append(np.sum(train_dataset.labels[:, i] == 0))
    elif args.loss_type in ['llr', 'llct']:
        clean_rate = 1


    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        model.train()

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].float().cuda(non_blocking=True)
            idx = pack['idx']
            x = model(img)

            optimizer.zero_grad()

            # loss = F.binary_cross_entropy_with_logits(x, label)

            if args.loss_type == 'an':
                loss = F.binary_cross_entropy_with_logits(x, label)
            elif args.loss_type == 'llcp':
                if ep == 0 or args.delta_rel == 0:
                    loss = F.binary_cross_entropy_with_logits(x, label)
                else:
                    k = math.ceil(label.shape[0] * label.shape[1] * args.delta_rel * 0.01)
                    loss_matrix = F.binary_cross_entropy_with_logits(x, label, reduction='none')
                    corrected_loss_matrix = F.binary_cross_entropy_with_logits(x, torch.logical_not(label).float(), reduction='none')
                    unobserved_loss = (label == 0).bool() * loss_matrix
                    topk_loss = torch.topk(unobserved_loss.flatten(), k).values[-1]
                    correction_idx = torch.where(unobserved_loss >= topk_loss)
                    loss = torch.where(unobserved_loss < topk_loss, loss_matrix, corrected_loss_matrix).mean()

                    train_dataset.labels[idx[correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0 # LL-Cp
                    llcp_count += len(correction_idx[0])
            elif args.loss_type == 'llr':
                if ep == 0 or args.delta_rel == 0:
                    loss = F.binary_cross_entropy_with_logits(x, label)
                else:
                    k = math.ceil(label.shape[0] * label.shape[1] * (1-clean_rate))
                    loss_matrix = F.binary_cross_entropy_with_logits(x, label, reduction='none')
                    zero_loss_matrix = torch.zeros_like(loss_matrix)
                    unobserved_loss = (label == 0).bool() * loss_matrix
                    topk_loss = torch.topk(unobserved_loss.flatten(), k).values[-1]
                    loss = torch.where(unobserved_loss < topk_loss, loss_matrix, zero_loss_matrix).mean()
            elif args.loss_type == 'llct':
                if ep == 0 or args.delta_rel == 0:
                    loss = F.binary_cross_entropy_with_logits(x, label)
                else:
                    k = math.ceil(label.shape[0] * label.shape[1] * (1-clean_rate))
                    loss_matrix = F.binary_cross_entropy_with_logits(x, label, reduction='none')
                    corrected_loss_matrix = F.binary_cross_entropy_with_logits(x, torch.logical_not(label).float(), reduction='none')
                    unobserved_loss = (label == 0).bool() * loss_matrix
                    topk_loss = torch.topk(unobserved_loss.flatten(), k).values[-1]
                    loss = torch.where(unobserved_loss < topk_loss, loss_matrix, corrected_loss_matrix).mean()

            elif args.loss_type == 'em':
                preds = torch.sigmoid(x)
                loss_mtx = torch.zeros_like(preds)
                loss_mtx[label == 1] = neg_log(preds[label == 1])
                loss_mtx[label == 0] = -0.1 * (
                        preds[label == 0] * neg_log(preds[label == 0]) +
                        (1 - preds[label == 0]) * neg_log(1 - preds[label == 0])
                    )
                loss = loss_mtx.mean()
            elif args.loss_type == 'emapl':
                preds = torch.sigmoid(x)
                loss_mtx = torch.zeros_like(preds)
                loss_mtx[label == 1] = neg_log(preds[label == 1])
                loss_mtx[label == 0] = -0.1 * (
                        preds[label == 0] * neg_log(preds[label == 0]) +
                        (1 - preds[label == 0]) * neg_log(1 - preds[label == 0])
                    )
                soft_label = -label[label < 0]
                loss_mtx[label < 0] = 0.9 * (
                        soft_label * neg_log(preds[label < 0]) +
                        (1 - soft_label) * neg_log(1 - preds[label < 0])
                )
                loss = loss_mtx.mean()
            elif args.loss_type == 'ls':
                preds = torch.sigmoid(x)
                loss_mtx = torch.zeros_like(preds)
                loss_mtx[label == 1] = 0.9 * neg_log(preds[label == 1]) + 0.1 * neg_log(1.0 - preds[label == 1])
                loss_mtx[label == 0] = 0.9 * neg_log(1.0 - preds[label == 0]) + 0.1 * neg_log(preds[label == 0])
                loss = loss_mtx.mean()

            elif args.loss_type == 'asl':
                x_sigmoid = torch.sigmoid(x)
                xs_pos = x_sigmoid
                xs_neg = 1 - x_sigmoid

                xs_neg = (xs_neg + 0.2).clamp(max=1)

                los_pos = label * torch.log(xs_pos.clamp(min=1e-8))
                los_neg = (1 - label) * torch.log(xs_neg.clamp(min=1e-8))
                loss_mtx = los_pos + los_neg

                torch.set_grad_enabled(False)
                pt0 = xs_pos * label
                pt1 = xs_neg * (1 - label)
                pt = pt0 + pt1 
                one_sided_gamma = 2 * (1 - label)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                torch.set_grad_enabled(True)
                loss_mtx *= one_sided_w

                loss = -loss_mtx.mean()

            elif args.loss_type == 'role':
                estimated_labels = model_g(idx)
                preds = torch.sigmoid(x)
                loss_mtx_pos_1 = torch.zeros_like(label)
                loss_mtx_pos_1[label==1] = neg_log(preds[label==1])
                estimated_labels_detached = estimated_labels.detach()
                loss_mtx_cross_1 = estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(1.0 - preds)

                reg_1 = expected_positive_regularizer(preds, expected_num_pos, norm='2') / (args.num_classes ** 2)

                loss_mtx_pos_2 = torch.zeros_like(label)
                loss_mtx_pos_2[label==1] = neg_log(estimated_labels[label==1])
                preds_detached = preds.detach()
                loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(1.0 - estimated_labels)

                reg_2 = expected_positive_regularizer(estimated_labels, expected_num_pos, norm='2') / (args.num_classes ** 2)

                reg_loss = 0.5 * (reg_1 + reg_2)
                loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
                loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)

                loss = (loss_mtx + reg_loss).mean()

            loss.backward()
            avg_meter.add({'loss1': loss.item()})


            optimizer.step()
            # if (optimizer.global_step-1)%100 == 0:
            if (optimizer.global_step-1) in [100, 200, 300, 400, 500]: # only need to check whether loss goes to nan
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                      
        # validate(model, val_data_loader)
        timer.reset_stage()

        if args.loss_type in ['llr', 'llct']:
            clean_rate -= args.delta_rel
        elif args.loss_type == 'emapl' and ep >= 5:
            #pseudo-labeling
            model.eval()
            total_preds = None
            total_idx = None
            with torch.no_grad():
                for step, pack in enumerate(train_data_loader):
                    img = pack['img']
                    img = img.cuda()
                    label = pack['label'].float().cuda(non_blocking=True)
                    idx = pack['idx']
                    x = model(img)
                    preds = torch.sigmoid(x)

                    if step == 0:
                        total_preds = preds.detach().cpu().numpy()
                        total_idx = idx.cpu().numpy()
                    else:
                        total_preds = np.vstack((preds.detach().cpu().numpy(), total_preds))
                        total_idx = np.hstack((idx.cpu().numpy(), total_idx))

            for i in range(total_preds.shape[1]):  # class-wise

                class_preds = total_preds[:, i]
                class_labels_obs = train_dataset.labels[:, i]
                class_labels_obs = class_labels_obs[total_idx]

                # select unlabel data:
                unlabel_class_preds = class_preds[class_labels_obs == 0]
                unlabel_class_idx = total_idx[class_labels_obs == 0]

                # select samples:
                neg_PL_num = int(0.9 * unlabel_num[i] / 5)
                sorted_idx_loc = np.argsort(unlabel_class_preds)  # ascending
                selected_idx_loc = sorted_idx_loc[:neg_PL_num]  # select indices

                # assgin soft labels:
                for loc in selected_idx_loc:
                    train_dataset.labels[unlabel_class_idx[loc], i] = -unlabel_class_preds[loc]

    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()

    if args.loss_type == 'llcp':    
        print(f'Total permanent correction count: {llcp_count}')
        np.save(os.path.join(args.work_space, 'corrected_train_labels_spl.npy'), train_dataset.labels)