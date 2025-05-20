# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import scipy


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR

from utils.tools import AverageMeter
from utils.tools import *
from .losses import *
from .PPCL import PPCL


from pdb import set_trace as st

def train(model, trainloader, writer_dict, epoch, args):
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_cls_loss = AverageMeter()
    ave_pcl_loss = AverageMeter()
    ave_pos = AverageMeter()
    ave_neg = AverageMeter()
    tic = time.time()

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    iters_per_epoch = len(trainloader)

    if args.loss_type=='focal':
        lossFunc=FocalLoss()
    elif args.loss_type=='ce':
        lossFunc=F.cross_entropy
    
    if args.ppcl:
        pcl_loss_func = PPCL(
            num_classes=args.num_classes, 
            embedding=args.hidden_size,  
            epoch=epoch,
            alpha=8, 
            pos_loss_weight=0.00001,  
            neg_loss_weight=0.000001,  
            args=args
        )


    # optimizer
    if args.ppcl:
        optimizer = torch.optim.AdamW([{'params':
                                        filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                        'lr': args.lr},
                                        {'params':
                                        pcl_loss_func.parameters(),
                                        'lr': args.lr}],
                                        lr=args.lr,
                                        # weight_decay=0.05 # 0.01
                                    )     
    else:
        optimizer = torch.optim.AdamW([{'params':
                                        filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                        'lr': args.lr}],
                                        lr=args.lr,
                                        # weight_decay=0.05 # 0.01
                                    )        
    lr_scheduler = CosineAnnealingLR(
        optimizer, 
        args.num_epochs, #
        eta_min=1e-6     # 1e-6
    )

    for i_iter, batch in enumerate(trainloader):

        if args.MA:

            images, prompts, labels = batch
            images = images.float().cuda()
            if isinstance(prompts, list):
                prompts = torch.stack(prompts).float().cuda()
            else:
                prompts = prompts.float().cuda()
            labels = labels.long().cuda()
            logits, features = model(images,prompts)

        else:
            images, labels = batch
            images = images.float().cuda()
            labels = labels.long().cuda()

            logits, features = model(images)


        cls_loss = lossFunc(logits, labels)
        if args.ppcl:
            pcl_loss_func.init_proxies(features, labels, logits)
            pos, neg = pcl_loss_func(features, labels)
            losses = cls_loss + pos + neg #pcl_loss
        else:
            losses = cls_loss            

        loss = losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        if args.ppcl:
            ave_cls_loss.update(cls_loss.mean().item())
            ave_pos.update(pos.mean().item())
            ave_neg.update(neg.mean().item())

        lr = optimizer.param_groups[0]["lr"]

        est_time=batch_time.average()*((args.num_epochs-epoch)*iters_per_epoch+iters_per_epoch-i_iter)/3600
        
        if args.ppcl:
            msg = f'Epoch: [{epoch}/{args.num_epochs}] Iter:[{i_iter}/{iters_per_epoch}], Time: {batch_time.average():.2f}, ERT: {est_time:.2f}, lr: {lr:.6f}, cls_loss: {ave_cls_loss.average():.6f}, pos: {ave_pos.average():.6f}, neg: {ave_neg.average():.6f}, Loss: {ave_loss.average():.6f}' 
        else:
            msg = f'Epoch: [{epoch}/{args.num_epochs}] Iter:[{i_iter}/{iters_per_epoch}], Time: {batch_time.average():.2f}, ERT: {est_time:.2f}, lr: {lr:.6f}, Loss: {ave_loss.average():.6f}'
        
        print('\r',toGreen(msg),end='')
        
    writer_dict['train_global_steps'] = global_steps + 1


    lr_scheduler.step()
        

# def validate(model, valloader, writer_dict, args=None):
#     model.eval()
#     ave_loss = AverageMeter()

#     if args.loss_type=='focal':
#         lossFunc=FocalLoss()
#     elif args.loss_type=='ce':
#         lossFunc=F.cross_entropy

#     with torch.no_grad():
#         gts,probs,predicts=[],[],[]
#         for _, batch in enumerate(valloader):
#             if args.MA:
#                 images, prompts, labels = batch
#                 images = images.float().cuda()
#                 prompts = prompts.float().cuda()
#                 labels = labels.long().cuda()
#                 logits = model(images,prompts)
#             else:
#                 images, labels = batch
#                 images = images.float().cuda()
#                 labels = labels.long().cuda()
#                 logits = model(images)

#             preds = torch.argmax(logits,dim=-1)
#             gts.append(labels.cpu().numpy())
#             probs.append(scipy.special.softmax(logits.cpu().numpy(),axis=-1))
#             predicts.append(preds.cpu().numpy())

#             losses = lossFunc(logits,labels)
#             loss = losses.mean()
#             ave_loss.update(loss.item())

#         # single label:
#         gts=np.hstack(gts) 
#         probs=np.concatenate(probs,axis=0)
#         predicts=np.hstack(predicts)

#         # multi labels
#         # gts=np.vstack(gts) 
#         # predicts=np.vstack(predicts)
#         accuracy, specificity, recall, f1, auc = mymetrics(gts,probs,predicts)

#     writer = writer_dict['writer']
#     global_steps = writer_dict['valid_global_steps']
#     # writer.add_scalar('valid/loss', ave_loss.average(), global_steps)
#     # writer.add_scalar('valid/accuracy', accuracy, global_steps)
#     # writer.add_scalar('valid/specificity', specificity, global_steps)
#     # writer.add_scalar('valid/recall', recall, global_steps)
#     # writer.add_scalar('valid/f1', f1, global_steps)
#     # writer.add_scalar('valid/auc', auc, global_steps)

#     writer_dict['valid_global_steps'] = global_steps + 1

#     return accuracy, specificity, recall, f1, auc


def test(model, testloader, writer_dict=None, args=None):
    model.eval()
    with torch.no_grad():
        gts,probs,predicts=[],[],[]
        for _, batch in tqdm(enumerate(testloader), total=len(testloader), desc="Testing"):
            if args.MA:
                images, prompts, labels = batch
                images = images.float().cuda()
                if isinstance(prompts, list):
                    prompts = torch.stack(prompts).float().cuda()
                else:
                    prompts = prompts.float().cuda()
                labels = labels.long().cuda()
                logits, _ = model(images,prompts,None)
              
            else:
                images, labels = batch
                images = images.float().cuda()
                labels = labels.long().cuda()
                logits, _ = model(images)

            preds = torch.argmax(logits,dim=-1)

            gts.append(labels.cpu().numpy())
            probs.append(scipy.special.softmax(logits.cpu().numpy(),axis=-1))
            predicts.append(preds.cpu().numpy())

            # losses = F.cross_entropy(logits,labels)
            # loss = losses.mean()

        # single label:
        gts=np.hstack(gts) 
        probs=np.concatenate(probs,axis=0)
        predicts=np.hstack(predicts)


        accuracy, specificity, recall, f1, auc = mymetrics(gts,probs,predicts)
        _, ovr_specificity, ovr_recall, ovr_f1, ovr_auc = mymetrics_without_avg(gts,probs,predicts)
        con_mat = confusion_matrix(gts, predicts)
        ovr_mat = np.array([ovr_specificity,ovr_recall,ovr_f1,ovr_auc]) 
        ovr_mat = np.round(ovr_mat*100,2)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['test_global_steps']
            writer_dict['test_global_steps'] = global_steps + 1


        return gts,predicts,probs,accuracy, specificity, recall, f1, auc, con_mat, ovr_mat
