# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:04:37 2023

@author: zheng
"""
import json
import os

import torch
from PIL import Image
import numpy as np

import lpips
import os
from guided_diffusion.evaluation import evaluation
from utils.config import Config
import argparse


def normalize(image, shape=(224, 224)):

    image = np.array(image.convert("RGB").resize(shape))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image)
    image = image * 2.0 - 1.0
    image = image[None].permute(0, 3, 1, 2)
    return image

def test_classifier():
    parser = argparse.ArgumentParser(description="UAEs evaluation")
    parser.add_argument('--use_specific_path',type= bool, default=False,)
    parser.add_argument('--specific_path',type=str, default = None)
    parser.add_argument('--config',type=str, default = "configs/imagenet_perturb.yaml")
    parser.add_argument('--input_image', type = str, default = "./imagenet-compatible/images/")
    args = parser.parse_args()
    config_file = args.config
    specific_path = args.specific_path
    use_specific_path = args.use_specific_path
    config = Config(default_config_file=config_file, use_argparse=False)

    if use_specific_path:
        evaluation(config,specific_path=specific_path)
        path = specific_path
    else:
        evaluation(config)
        temp_path = os.path.join(config.outdir, config.target_model)
        path = os.path.join(temp_path, 'samples')

    gt_path = args.input_image
    images = os.listdir(path)
    lp = []
    lpips_model = lpips.LPIPS(net="alex").cuda()
    for image in images:
        image_path = os.path.join(path, image)
        im = Image.open(image_path)

        img = normalize(im)
        gt_image = str(int(os.path.splitext(image)[0].split('_')[1]))+'.png'

        gt = os.path.join(gt_path, gt_image)
        im_gt = Image.open(gt)
        img_gt = normalize(im_gt)

        img = img.cuda()
        img_gt = img_gt.cuda()
        l = lpips_model(img, img_gt)
        lp.append(l.cpu().detach().numpy())

    print("average lpips is {}".format(np.sum(lp)/len(lp)))

if __name__ == '__main__':

    test_classifier()
