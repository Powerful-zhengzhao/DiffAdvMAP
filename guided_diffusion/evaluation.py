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
# import dill
# import matplotlib.pyplot as plt
import lpips
# from robustbench.utils import load_model
import os
import timm
from torch_nets import (
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
from art.estimators.classification import PyTorchClassifier
import pytorch_fid.fid_score as fid_score
from Finegrained_model import model as otherModel

def model_selection(name,batch_size = 50):
    if name == "Pure_Resnet50":
        model = models.resnet50(pretrained=True)
    elif name == "swin-transformer":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "mobilenet-v2":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "inception-v3":
        model = models.inception_v3(pretrained=True)
    elif name == 'vgg':
        model = models.vgg19(pretrained=True)
    elif name == 'convnext':
        model = models.convnext_base(pretrained=True)
    elif name == "vit":
        model = models.vit_b_16(pretrained=True)
    elif name == "deit-b":
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=True,
            # pretrained_cfg_overlay = dict(file='C:/Users/zheng/.cache/huggingface/hub/models--timm--deit_base_patch16_224.fb_in1k'),
        )
    elif name == "deit-s":
        model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True
        )
    elif name == "mixer-b":
        model = timm.create_model(
            'mixer_b16_224',
            pretrained=True
        )
    elif name == "mixer-l":
        model = timm.create_model(
            'mixer_l16_224',
            pretrained=True
        )
    elif name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'cubResnet50':
        model = otherModel.CUB()[0]
    elif name == 'cubSEResnet154':
        model = otherModel.CUB()[1]
    elif name == 'cubSEResnet101':
        model = otherModel.CUB()[2]
    elif name == 'carResnet50':
        model = otherModel.CAR()[0]
    elif name == 'carSEResnet154':
        model = otherModel.CAR()[1]
    elif name == 'carSEResnet101':
        model = otherModel.CAR()[2]
    else:
        raise NotImplementedError("No such model!")
    return model.cuda()

models_transfer_name = ["inception-v3","Pure_Resnet50", "mobilenet-v2", 'vgg','convnext','vit','swin-transformer',"deit-b","deit-s","mixer-b","mixer-l"]
timm_models_name = ["vit","deit-b","deit-s","mixer-b","mixer-l"]
robust_models_name = ['tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3',
                                'tf2torch_ens4_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2']
cub_models_name = ["cubResnet50", "cubSEResnet154", "cubSEResnet101"]
car_models_name = ["carResnet50", "carSEResnet154", "carSEResnet101"]


def normalize(image,adv_path,shape=(224, 224)):

    image = np.array(image.convert("RGB").resize(shape))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    if 'imagenetsc' in adv_path:
        image = image[:,:,16:240,16:240]

    return image
def evaluation(config,adversarial_trained,specific_path = None):
    res = 224
    nb_classes = 1000
    all_accuracy = []
    if specific_path == None:
        outpath = os.path.join(config.outdir, config.target_model)
        adv_path = os.path.join(outpath, "samples")
        target_model = config.target_model
    else:
        adv_path = specific_path
        target_model = None

    if 'imagenet' in adv_path:
        print('load data from {}'.format(adv_path))
        with open('./imagenet-compatible/labels.txt', "r") as f:
            label = []
            for i in f.readlines():
                label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
            label = np.array(label)
            f.close()

        image_names = os.listdir(adv_path)
        image_names.sort(key=lambda x: int(x.split('.')[0]))
        # print(image_names)
        for name in models_transfer_name:
            if 'imagenetsc' in adv_path:
                res = 256
            images = []
            origin_labels = []
            model = model_selection(name)
            model.eval()
            f_model = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=nn.CrossEntropyLoss(),
                input_shape=(3, res, res),
                nb_classes=nb_classes,
                preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) if "mixer" in name else (
                    np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
                device_type='gpu',
            )
            for image_name in image_names:
                image_path = os.path.join(adv_path,image_name)
                im = Image.open(image_path)
                img = normalize(im,adv_path,shape=(res,res))
                images.append(img)
                origin_labels.append(int(os.path.splitext(image_name)[0].split('_')[0]))

            images = np.concatenate(images)
            origin_labels = np.array(origin_labels)
            adv_pred = f_model.predict(images,batch_size = 50)
            accuracy = np.sum(np.argmax(adv_pred, axis=1) != origin_labels) / len(origin_labels)

            print("Attack success rate on {}: {}%".format(name, accuracy * 100))
            if (specific_path is None and name != target_model) or (specific_path is not None and name not in specific_path):
                all_accuracy.append(accuracy)

        all_accuracy = np.array(all_accuracy)
        print("Mean success rate: {}%".format(np.mean(all_accuracy) * 100))

        fid = fid_score.main(adv_path, "imagenet_compatible")
        print("\n*********fid: {}********".format(fid))

    elif 'cub' in adv_path:
        print('load data from {}'.format(adv_path))
        with open('./CUB_200_2011/labels.txt', "r") as f:
            label = []
            for i in f.readlines():
                label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
            label = np.array(label)
            f.close()

        image_names = os.listdir(adv_path)
        image_names.sort(key=lambda x: int(x.split('.')[0]))
        for name in cub_models_name:
            images = []
            origin_labels = []
            model = model_selection(name)
            model.eval()
            f_model = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=nn.CrossEntropyLoss(),
                input_shape=(3, res, res),
                nb_classes=nb_classes,
                preprocessing=(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
                device_type='gpu',
            )
            for image_name in image_names:
                image_path = os.path.join(adv_path, image_name)
                im = Image.open(image_path)
                img = normalize(im, image_name)
                images.append(img)
                origin_labels.append(int(os.path.splitext(image_name)[0].split('_')[0]))

            images = np.concatenate(images)
            origin_labels = np.array(origin_labels)
            adv_pred = f_model.predict(images, batch_size=50)

            accuracy = np.sum(np.argmax(adv_pred, axis=1) != origin_labels) / len(origin_labels)
            print("Attack success rate on {}: {}%".format(name, accuracy * 100))
            if (specific_path is None and name != target_model) or (specific_path is not None and name not in specific_path):
                all_accuracy.append(accuracy)

        all_accuracy = np.array(all_accuracy)
        print("Mean success rate: {}%".format(np.mean(all_accuracy) * 100))

        fid = fid_score.main(adv_path, "cub_200_2011")
        print("\n*********fid: {}********".format(fid))

    elif 'car' in adv_path:
        print('load data from {}'.format(adv_path))
        with open('./Stanford_Car/labels.txt', "r") as f:
            label = []
            for i in f.readlines():
                label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
            label = np.array(label)
            f.close()

        image_names = os.listdir(adv_path)
        image_names.sort(key=lambda x: int(x.split('.')[0]))
        # print(image_names)
        for name in car_models_name:
            images = []
            origin_labels = []
            model = model_selection(name)
            model.eval()
            f_model = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=nn.CrossEntropyLoss(),
                input_shape=(3, res, res),
                nb_classes=nb_classes,
                preprocessing=(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
                device_type='gpu',
            )
            for image_name in image_names:
                image_path = os.path.join(adv_path, image_name)
                im = Image.open(image_path)
                img = normalize(im, image_name)
                images.append(img)
                origin_labels.append(int(os.path.splitext(image_name)[0].split('_')[0]))

            images = np.concatenate(images)
            origin_labels = np.array(origin_labels)
            adv_pred = f_model.predict(images, batch_size=50)

            accuracy = np.sum(np.argmax(adv_pred, axis=1) != origin_labels) / len(origin_labels)
            print("Attack success rate on {}: {}%".format(name, accuracy * 100))
            if (specific_path is None and name != target_model) or (specific_path is not None and name not in specific_path):
                all_accuracy.append(accuracy)

        all_accuracy = np.array(all_accuracy)
        print("Mean success rate: {}%".format(np.mean(all_accuracy) * 100))

        fid = fid_score.main(adv_path, "standford_car")
        print("\n*********fid: {}********".format(fid))



    if adversarial_trained == True:
        for name in robust_models_name:
            images = []
            origin_labels = []
            model = model_selection(name)
            model.eval()
            f_model = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=nn.CrossEntropyLoss(),
                input_shape=(3, res, res),
                nb_classes=nb_classes,
                preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) if "adv" in name else (
                    np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
                device_type='gpu',
            )
            for image_name in image_names:
                image_path = os.path.join(adv_path, image_name)
                im = Image.open(image_path)
                img = normalize(im, name)
                images.append(img)
                origin_labels.append(int(os.path.splitext(image_name)[0].split('_')[0]))

            images = np.concatenate(images)
            origin_labels = np.array(origin_labels)
            adv_pred = f_model.predict(images, batch_size=50)
            accuracy = np.sum(np.argmax(adv_pred, axis=1)-1 != origin_labels) / len(origin_labels)
            print("Attack success rate on {}: {}%".format(name, accuracy * 100))

