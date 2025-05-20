# https://huggingface.co/MahmoodLab/UNI
import timm
from torchvision import transforms
import torch
import torch.nn as nn
import os

from . import vision_transformer as vits
    
    
def get_uni_trans():
    transform = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        
    return transform



def get_uni_model(device, args=None):
    # model = timm.create_model(
    #     "vit_large_patch16_224", img_size=224, num_classes=0, dynamic_img_size=True, patch_size=16,  init_values=1e-5
    # )
    model = vits.vit_large_patch16_224(img_size=224, dynamic_img_size=True, patch_size=16, init_values=1e-5, args=args)

    # model.load_state_dict(torch.load('model/ckpts/uni.bin', map_location="cpu"), strict=True)

    pretrained_dict = torch.load('model/ckpts/uni.bin')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     print('=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    msg = model.load_state_dict(model_dict)
    print('=> loading from pretrained model {}'.format(msg))
    

    return model.to(device)