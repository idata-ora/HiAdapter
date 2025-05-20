import os

import cv2
import numpy as np
from PIL import Image
import random
import glob

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from imgaug import augmenters as iaa
from skimage.color import rgb2hed
from utils.utils import *

from skimage import measure
import matplotlib.pyplot as plt

from pdb import set_trace as st





def _read_array_image(path,resize,float_type=True,rgb=False,pil_format=False):
    
    if pil_format:
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error reading image from {path}: {e}")
    else:
        img=cv2.imread(path,-1)
        if float_type: img=img.astype('float32')
        img=cv2.resize(img,(resize[0],resize[1]))
        if rgb: img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class myTileDataset(torch.utils.data.Dataset):
    def __init__(self,XY,dataDir,h=256,w=256,is_training=False,return_path=False,augment=False,
                transform=None,args=None):

        self.dataDir=dataDir
        self.is_training=is_training
        self.return_path=return_path
        self.augment=augment
        self.args=args
        self.XY_list = XY
        self.h = h
        self.w = w
        
        base_transforms_list=transform

                           
        self.augPools = [
            transforms.RandomHorizontalFlip(p=0.5),      
            transforms.RandomVerticalFlip(p=0.5),        
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.GaussianBlur((3,7), sigma=(0.1, 1.5))
        ]          
        # 0-1-2
        # 0-1-2-3-4
        # 0-1-2-3-4-5


        aug_transforms_list=[]
        if self.augment and self.args.augment_index!='none':
            for index in self.args.augment_index.split('-'):
                aug_transforms_list.append(self.augPools[int(index)])


        if self.is_training:
            self.transforms=transforms.Compose(aug_transforms_list+base_transforms_list)
        else:
            self.transforms=transforms.Compose(base_transforms_list)


        
    def __len__(self):
        return len(self.XY_list)

    def get_labels(self):
        labels = []
        for item in self.XY_list:
            ss=item.split('.')
            label=int(ss[2])
            labels.append(label)

        return labels

    def __getitem__(self, idx):

        
        ss=self.XY_list[idx].split('.')
        _path1,subpath=ss[0]+".png",ss[1] # 'jxl1362155_patch_0_9984_9472.png', '2'

 
        class_subfoler=_path1.split(os.sep)[0]
        path1=os.path.join(self.dataDir,subpath,_path1) 

        label=int(subpath)


        img=_read_array_image(path1,[self.h, self.w],pil_format=True)
        img=self.transforms(img)

        if self.args.MA:
            MA = []
            for prompt_path in self.args.MA_folder.split('-'):
                tmp=_read_array_image(os.path.join(self.dataDir,prompt_path,_path1),[self.h, self.w],pil_format=True) 
                tmp=self.transforms(tmp)
                MA.append(tmp)
            return img, MA, label


        if self.return_path:
            return img, label, path1
        else:
            return img, label