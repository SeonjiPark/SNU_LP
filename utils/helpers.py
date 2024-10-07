import torch.nn.init as init
import torch
import os
import numpy as np
  
def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def initialize_model(model):
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    return model

def filter_parameter(model):
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    
    return filtered_parameters

def encode_province(text, province, province_replace):

    for idx in range(len(province)):
        prov = province[idx]
        if prov in text:
            text = text.replace(prov, province_replace[idx])

    return text

def decode_province(text, province, province_replace):

    for idx in range(len(province)):
        prov = province_replace[idx]
        if prov in text:
            text = text.replace(prov, province[idx])
    return text

def enhance_edges(imgs, edge_amount):
    sobel_x = torch.Tensor([[[-1, 0, 1], 
                             [-2, 0, 2], 
                             [-1, 0, 1]]]).to(imgs.device)

    sobel_y = torch.Tensor([[[-1, -2, -1], 
                             [ 0,  0,  0], 
                             [ 1,  2,  1]]]).to(imgs.device)

    sobel_x = sobel_x.expand(imgs.size(1), 1, 3, 3)
    sobel_y = sobel_y.expand(imgs.size(1), 1, 3, 3)

    edges_x = torch.nn.functional.conv2d(imgs, sobel_x, padding=1, groups=imgs.size(1))
    edges_y = torch.nn.functional.conv2d(imgs, sobel_y, padding=1, groups=imgs.size(1))
    
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    
    enhanced_imgs = imgs + edge_amount*edges
    
    enhanced_imgs = (enhanced_imgs - enhanced_imgs.min()) / (enhanced_imgs.max() - enhanced_imgs.min())

    return enhanced_imgs