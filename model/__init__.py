import torch
import numpy as np


def get_model(model_name, device, args=None):
    """
    Args:
        model_name (str): the name of the requried model
        device (torch.device): device, e.g. 'cuda'
    Returns:
        nn.Module: model
    """

    if model_name.lower() == 'uni':
        from model.uni import get_uni_model
        model = get_uni_model(device, args=args)

    return model


def get_custom_transformer(model_name, args=None):
    """
    Args:
        model_name (str): the name of model
    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    
    if model_name.lower() == 'uni':
        from model.uni import get_uni_trans
        custom_trans = get_uni_trans()

    return custom_trans
