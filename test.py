import argparse
import os
import pprint
import shutil
import sys
import importlib

import logging
import time
import timeit
from pathlib import Path
import glob

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

from myDatasets.myPromptAugTileDataset import myTileDataset
from core.function import train, test #validate
from utils.params import *
from utils.utils import *
from utils.spatial_statistics import *

from model import get_model, get_custom_transformer


from pdb import set_trace as st

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print('----------------------test----------------------')
    args=parse_args()

    data_dir = ''
    file_list_folder = ''
    args.model_name = 'uni'
    RESUME_FOLDER = ''


    with open(os.path.join(RESUME_FOLDER,'commandline_args.json'), 'r') as json_file:
        args_json = json.load(json_file)
        print(toGreen(f'updating args from commandline_args.json'))
        for key, value in args_json.items():
            setattr(args, key, value)

    CKPT_PATH=os.path.join(RESUME_FOLDER, 'best.pth')


    gpus = [ii for ii in range(args.gpus)]

    setup_seed(args.seed)


    _x_test = [os.path.join(file_list_folder, 'test.txt')]

    x_test =readlines_from_txt(_x_test)
    x_test_tileNames=[x.split('.')[0] for x in x_test]

    custom_transformer = get_custom_transformer(args.model_name, args=args)


    test_dataset=myTileDataset(x_test,data_dir,h=args.img_h,w=args.img_w,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props,transform=custom_transformer,args=args)
    

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

    ## load the ckpt dict
    model_state_file = CKPT_PATH
    print(toGreen(f'loading ckpt {CKPT_PATH}'))
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        print(toCyan('=> loading {} from pretrained model'.format(k)))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

   
    gts, predicts, probs, accuracy, specificity, recall, f1, auc, con_mat, ovr_mat = test(model, testloader, writer_dict=None, args=args)
    test_result = np.concatenate([np.expand_dims(gts,-1),np.expand_dims(predicts,-1),probs],axis=-1)
    test_result_excel={'tileNames':x_test_tileNames,'gts':gts.tolist(),'predicts':predicts.tolist()}
    test_msg = f'{args.model_name}:Accuracy:{accuracy*100:.3f}, Specificity:{specificity*100:.3f}, Recall:{recall*100:.3f}, F1:{f1*100:.3f}, AUC:{auc*100:.3f}'

    print(toRed(test_msg))
    print('Multi-class one-vs-rest result\t')
    print(str(ovr_mat)+'\n')
    print('Confusion Matrix\t')
    print(str(con_mat)+'\n')


    # save
    results_file = os.path.join(RESUME_FOLDER, 'best_pth_test_results.txt')
    with open(results_file, 'w') as f:
        f.write('\t')
        f.write('---'*5 + '\n')
        f.write(RESUME_FOLDER+'\n')
        f.write(test_msg+'\n')
        f.write('Multi-class one-vs-rest result\n')
        f.write(str(ovr_mat)+'\n')
        f.write('Confusion Matrix\n')
        f.write(str(con_mat)+'\n')
        f.close()
    # np.save(os.path.join(RESUME_FOLDER, 'test.npy'),test_result)
    df = pd.DataFrame.from_dict(test_result_excel)
    df.to_excel(os.path.join(RESUME_FOLDER, 'test.xlsx'))

    print(f"Test results saved to {results_file}")

       

if __name__ == '__main__':
    main()
