import json
import os

import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
import matplotlib.pyplot as plt
from metrics import SSIM
import lpips
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification, BeitFeatureExtractor, BeitForImageClassification, MobileViTForImageClassification, SwinForImageClassification
import os
import timm
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
im_size = 256

vit = False
compute_metrics= False

def normalize(image, shape=(256, 256)):

    image = np.array(image.convert("RGB").resize(shape))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image)
    if vit==True:
        mean = torch.as_tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        # mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device)
        # std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device)
        # image = image[:, :, :].sub(mean).div(std)
        image = image[None]
        trans = transforms.CenterCrop(224)
        image = image.permute(0, 3, 1, 2)
        image = trans(image)
    else:
        # mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device)
        # std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device)
        # # mean = torch.as_tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        # # std = torch.as_tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        # if not compute_metrics:
        #     image = image[:, :, :].sub(mean).div(std)
        # else:
        #     image = image * 2.0 - 1.0
        image = image[None].permute(0, 3, 1, 2)
        # trans = transforms.Resize((299,299))
        # trans = transforms.CenterCrop(224)
        # image = trans(image)
    return image
import shutil
gt_path = 'D:/github/imagenet'
path = 'D:/github/DiffAttack/output1/'
save_path = 'D:/github/DiffAdvMAP_results/DiffAttack/convnext/'
# path = 'D:/github/imagenet-compatible/images'
# save_path = 'D:/github/imagenet-compatible-256'
images = os.listdir(path)
for image in images:
    if 'adv' in image:
        image_path = os.path.join(path,image)
        new_name = str(int(os.path.splitext(image)[0].split('_')[0]))+'.png'
        new_path = os.path.join(save_path,new_name)
        shutil.move(image_path,new_path)

