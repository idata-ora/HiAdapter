import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import glob
import pandas as pd
import importlib

import re

import numpy as np
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR

from core.function import train, test #validate
from utils.params import *
from utils.utils import *
from myDatasets.myTileDataset import myTileDataset

from model import get_model, get_custom_transformer
from PIL import Image

from pdb import set_trace as st

PYTORCH_CUDA_ALLOC_CONF=True

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_print(message, file_path=None):
    print(message)  
    with open(file_path, 'a') as f:  
        f.write(message + '\n')


from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset):

    if hasattr(dataset, 'targets'):  
        labels = torch.tensor(dataset.targets)
    else:  
        labels = []
        for i in range(len(dataset)):
            label = dataset[i][-1]  
            labels.append(label)
        labels = torch.tensor(labels)


    class_counts = torch.bincount(labels)
    num_classes = len(class_counts)
    

    class_weights = 1. / (class_counts.float() + 1e-3)
    

    sample_weights = class_weights[labels]


    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  
        replacement=True  
    )
    return sampler



def main():
    args = parse_args()


    # 5-fold cross validation
    for ffold in args.folds.split('-'):
        snapshot_dir = args.snapshot_dir
        args.modelfold_name = f'F{ffold}-seed{args.seed}'
        
        if args.MA:
            if args.ppcl:
                if args.SI:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + args.MA_folder + '-' + args.SI_folder + "-ppcl"  + "/" + args.modelfold_name + '-')
                else:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + args.MA_folder + "-ppcl/" + args.modelfold_name + '-')
                args.save_proxies = snapshot_dir
            else:
                if args.SI:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + args.MA_folder + "-" + args.SI_folder + "/" + args.modelfold_name + '-')
                else:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + args.MA_folder + "/" + args.modelfold_name + '-')
        else:
            if args.ppcl:
                if args.SI:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + args.SI_folder + "-ppcl" + "/" + args.modelfold_name + '-')
                else:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + "ppcl/" + args.modelfold_name + '-')
                args.save_proxies = snapshot_dir
            else:
                if args.SI:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + args.SI_folder + "/" + args.modelfold_name + '-')
                else:
                    snapshot_dir = snapshot_dir.replace(
                        'model/', args.data_name + '/' + args.model_name +'/' + "-" + "/" + args.modelfold_name + '-')


        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        print(toMagenta(snapshot_dir))

        # saving args into txt file for recording
        with open(os.path.join(snapshot_dir,'commandline_args.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)


        writer_dict = {
            'writer': SummaryWriter(snapshot_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
            'test_global_steps':0
        }

        # cudnn related setting
        gpus = [ii for ii in range(args.gpus)]

        setup_seed(args.seed)

        _x_train = [os.path.join(args.file_list_folder, 'train_fold_'+str(ffold)+'.txt')]
        _x_val = [os.path.join(args.file_list_folder, 'val_fold_'+str(ffold)+'.txt')]
        _x_test = [os.path.join(args.file_list_folder, 'test_fold_'+str(ffold)+'.txt')]
        

        x_train = readlines_from_txt(_x_train)
        x_val = readlines_from_txt(_x_val)
        x_test = readlines_from_txt(_x_test)

        x_val_tileNames=[x.split('.')[0] for x in x_val]
        x_test_tileNames=[x.split('.')[0] for x in x_test]

        ##todo
        custom_transformer = get_custom_transformer(args.model_name, args=args)

        train_dataset=myTileDataset(x_train,args.data_dir,h=args.img_h,w=args.img_w,is_training=True,augment=args.augment,transform=custom_transformer, args=args)
        val_dataset=myTileDataset(x_val,args.data_dir,h=args.img_h,w=args.img_w,transform=custom_transformer,args=args)
        test_dataset=myTileDataset(x_test,args.data_dir,h=args.img_h,w=args.img_w,transform=custom_transformer,args=args)
        

        ##todo
        if args.sampler:
            train_sampler = create_balanced_sampler(train_dataset)
            trainloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                # prefetch_factor=2, ##todo
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
                sampler=train_sampler ##
                )
        else:
            trainloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                # prefetch_factor=2, ##todo
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True
                )


        valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size_for_test,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_for_test,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)



        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Device:{}, GPU Count:{}, use GPU:{}'.format(device.type, torch.cuda.device_count(), args.gpus))
        model = get_model(args.model_name, device, args=args) #torch.cuda.device_count()
        

        print(toRed(f'[model]-{args.model_name}'))



        if args.frozen:
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
        # trainable
        for name, param in model.head.named_parameters():
            param.requires_grad = True
        if args.SI:
            for name, param in model.MulLayer.named_parameters():
                param.requires_grad = True
            if args.cdeep_fusion:
                for name, param in model.cfusion_layer.named_parameters():
                    param.requires_grad = True
        if args.MA:
            for name, param in model.prompt_generator.named_parameters():
                param.requires_grad = True
        ## check
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'*** requires_grad: {name}') 
        
        model = nn.DataParallel(model, device_ids=gpus).cuda() 


        best_val_f1 = 0
        best_val_msg=''
        best_val_con_mat=0
        best_val_ovr_mat=0
    

        ## load the ckpt dict
        if args.ckpt_path:
            model_state_file = args.ckpt_path
            pretrained_dict = torch.load(model_state_file)
            pretrained_dict = {f"module.{k}": v for k, v in pretrained_dict.items()} ##todo
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                print(toCyan('=> loading {} from pretrained model'.format(k)))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)


        # write final results into specific record.txt
        subf = open(os.path.join(snapshot_dir, 'record.txt'),'a+')
        patience = 10  
        early_stopping_counter = 0  
        for epoch in range(args.start_epoch, args.num_epochs):
            train(model, trainloader, writer_dict, epoch, args) 


            print('\t')
            print('---model_name---')
            print(args.modelfold_name)

            
            gts, predicts, probs, accuracy, specificity, recall, f1, auc, con_mat, ovr_mat = test(model, valloader, writer_dict, args)

            val_result = np.concatenate([np.expand_dims(gts,-1),np.expand_dims(predicts,-1),probs],axis=-1)

            val_result_excel={'tileNames':x_val_tileNames,'gts':gts.tolist(),'predicts':predicts.tolist()}

            val_msg = f'{args.modelfold_name}: Epoch:{epoch}, Accuracy:{accuracy*100:.3f}, Specificity:{specificity*100:.3f}, Recall:{recall*100:.3f}, F1:{f1*100:.3f}, AUC:{auc*100:.3f}'

           
            # custom_print('val result: '+ val_msg, snapshot_dir + '/log.txt')



            if f1 > best_val_f1 and epoch > args.record_epoch:
                best_val_f1 = f1
                best_val_msg=val_msg
                best_val_con_mat=con_mat
                best_val_ovr_mat=ovr_mat
                print(toMagenta('Best is: '+best_val_msg))
                print('Multi-class one-vs-rest result\t')
                print(str(ovr_mat)+'\n')
                print('Confusion Matrix\t')
                print(str(con_mat)+'\n')
                torch.save(model.module.state_dict(),
                        os.path.join(snapshot_dir, 'best.pth'))
                
                # np.save(os.path.join(snapshot_dir, f'T{ffold}.npy'),val_result)
                # df = pd.DataFrame.from_dict(val_result_excel)
                # df.to_excel(os.path.join(snapshot_dir, f'T{ffold}.xlsx'))


                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Early stopping counter: {early_stopping_counter}/{patience}")
         
                
            subf.write('\t')
            subf.write('---'*5 + '\n')
            # subf.write('validation results ...\n')
            # subf.write(val_msg+'\n')
            subf.write('val results ...\n')
            subf.write(val_msg+'\n')
            subf.write('Multi-class one-vs-rest result\n')
            subf.write(str(ovr_mat)+'\n')
            subf.write('Confusion Matrix\n')
            subf.write(str(con_mat)+'\n')
            subf.write('Best val results ...\n')
            subf.write(best_val_msg+'\n')


            if (early_stopping_counter >= patience):
                print(f"Early stopping triggered after {epoch} epochs.")
                custom_print(f"Early stopping triggered after {epoch} epochs.", snapshot_dir + '/log.txt')
                break


        f = open(snapshot_dir + '/best_pth_val_results.txt', 'a+')
        f.write('\t')
        f.write('---'*5 + '\n')
        f.write(snapshot_dir+'\n')
        f.write(best_val_msg+'\n')
        f.write('Multi-class one-vs-rest result\n')
        f.write(str(best_val_ovr_mat)+'\n')
        f.write('Confusion Matrix\n')
        f.write(str(best_val_con_mat)+'\n')
        f.close()

    
        torch.save(model.module.state_dict(),
                os.path.join(snapshot_dir, 'final_state.pth'))


        ## test
        # model.load_state_dict(torch.load(os.path.join(snapshot_dir, 'best.pth'), map_location=torch.device('cuda')))
        best_pth = torch.load(os.path.join(snapshot_dir, 'best.pth'))
        fixed_best_pth = {f"module.{k}": v for k, v in best_pth.items()}
        model_dict = model.state_dict()
        fixed_best_pth = {k: v for k, v in fixed_best_pth.items()
                        if k in model_dict.keys()}
        # for k, _ in fixed_best_pth.items():
        #     print('=> loading {} from pretrained model'.format(k))
        model_dict.update(fixed_best_pth)
        model.load_state_dict(model_dict)
        msg = model.load_state_dict(model_dict)
        print('=> loading from pretrained model {}'.format(msg))


        gts, predicts, probs, accuracy, specificity, recall, f1, auc, con_mat, ovr_mat = test(model, testloader, writer_dict, args)
        test_result = np.concatenate([np.expand_dims(gts,-1),np.expand_dims(predicts,-1),probs],axis=-1)
        test_result_excel={'tileNames':x_test_tileNames,'gts':gts.tolist(),'predicts':predicts.tolist()}
        test_msg = f'{args.modelfold_name}: Epoch:{epoch}, Accuracy:{accuracy*100:.3f}, Specificity:{specificity*100:.3f}, Recall:{recall*100:.3f}, F1:{f1*100:.3f}, AUC:{auc*100:.3f}'
        print(toMagenta('Best test is: ' + test_msg))

        # save
        results_file = os.path.join(snapshot_dir, 'best_pth_test_results.txt')
        with open(results_file, 'w') as f:
            f.write('\t')
            f.write('---'*5 + '\n')
            f.write(snapshot_dir+'\n')
            f.write(test_msg+'\n')
            f.write('Multi-class one-vs-rest result\n')
            f.write(str(ovr_mat)+'\n')
            f.write('Confusion Matrix\n')
            f.write(str(con_mat)+'\n')
            f.close()
        # np.save(os.path.join(snapshot_dir, f'test_T{ffold}.npy'), test_result)
        df = pd.DataFrame.from_dict(test_result_excel)
        df.to_excel(os.path.join(snapshot_dir, f'test_T{ffold}.xlsx'))

        print(f"Test results saved to {results_file}")




if __name__ == '__main__':
    main()

