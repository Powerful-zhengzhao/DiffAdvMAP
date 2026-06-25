import random
import torch
import os
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate
import math
import torch.optim as optim
import kornia
from PIL import Image


def normalize(image, shape=(224, 224)):
    """
    Given an PIL image, resize it and normalize each pixel into [-1, 1].
    Args:
        image: image to be normalized, PIL.Image
        shape: the desired shape of the image

    Returns: the normalized image

    """
    image = np.array(image.convert("RGB").resize(shape))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image * 2.0 - 1.0
    return image


class VGGFeatureExtractor(nn.Module):
    def __init__(self, content_layers, style_layers):
        super().__init__()
        # 加载预训练的VGG19模型
        vgg = models.vgg19(pretrained=True).features

        self.content_layers = content_layers
        self.style_layers = style_layers

        # 将VGG模型切分为多个模块，只保留到我们需要的最后一层
        self.model = nn.Sequential()
        i = 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{i}"
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            self.model.add_module(name, layer)

            # 如果已经添加了所有需要的层，就停止
            if name in content_layers and name in style_layers and name == max(content_layers + style_layers,
                                                                               key=lambda n: int(n.split('_')[1])):
                break

    def forward(self, x):
        content_features = []
        style_features = []
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in self.content_layers:
                content_features.append(x)
            if name in self.style_layers:
                style_features.append(x)
        return content_features, style_features

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    # 标准化Gram矩阵
    return G.div(b * c * h * w)
class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_graph=retain_variables)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_graph=retain_variables)
        return self.loss


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.to(content_img.device)
    cnn.eval()

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    # if use_cuda:
    #     model = model.cuda()
    #     gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses


def style_transfer(x, x_refer, mask, content_w, style_w, num_iters=300):
    model, style_losses, content_losses = get_style_model_and_losses(x_refer, x, style_w, content_w)

    x = x.clone()
    input_param = nn.Parameter(x)

    # optimizer =  optim.SGD([input_param], lr=0.01, momentum=0.9)
    optimizer = optim.Adam([input_param], lr=0.01, )

    run = [0]

    while run[0] < num_iters:
        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            input_param_new = input_param
            model(input_param_new)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    return input_param
def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image

    # check after max pooling layers

    # outputs and corresponding layers
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation output
                  '28': 'conv5_1'}

    ## -- do not need to change the code below this line -- ##
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    batch_size, d, h, w = tensor.size()

    tensor = tensor.view(d, h * w)  # can be just d,h*w

    gram = torch.mm(tensor, tensor.t())  # multiply with transpose

    ## get the batch_size, depth, height, and width of the Tensor
    ## reshape it, so we're multiplying the features for each channel
    ## calculate the gram matrix

    return gram


def get_style_loss(image,style_image):
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.to(image.device)
    cnn.eval()

    content_features = get_features(image, cnn)
    style_features = get_features(style_image, cnn)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = image.clone().requires_grad_(True).to(image.device)
    target_features = get_features(target, cnn)

    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    content_weight = 1
    style_loss = 0
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = content_features[layer]

        ## TODO: Calculate the target gram matrix
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape

        # get the "style" style representation
        style_gram = style_grams[layer]

        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    total_loss =  1e3 * style_loss

    return total_loss


class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_hook_fn):
        super().__init__()
        self.model = model
        self.feature_extractor = layer_hook_fn(self.model)

    def forward(self, x):
        return self.feature_extractor(x)

def get_vgg19_features(model):
    return nn.Sequential(*list(model.features.children())[:36])

def get_resnet50_features(model):
    return nn.Sequential(*list(model.children())[:-3])

def get_inceptionv3_features(model):
    feature_layers = nn.Sequential(
        model.Conv2d_1a_3x3,
        model.Conv2d_2a_3x3,
        model.Conv2d_2b_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2),
        model.Conv2d_3b_1x1,
        model.Conv2d_4a_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2),
        model.Mixed_5b,
        model.Mixed_5c,
        model.Mixed_5d,
        model.Mixed_6a,
        model.Mixed_6b,
        model.Mixed_6c,
        model.Mixed_6d,
        model.Mixed_6e,
        model.Mixed_7a,
        model.Mixed_7b,
        model.Mixed_7c
    )
    return feature_layers

def get_mobilenetv2_features(model):
    return model.features

def get_convnext_base_features(model):
    return nn.Sequential(*list(model.features.children())[:7])

def get_swin_b_features(model):
    class SwinFeatureExtractor(nn.Module):
        def __init__(self, swin_model):
            super().__init__()
            self.features = swin_model.features
            self.norm = swin_model.norm
        def forward(self, x):
            x = self.features(x)
            x = self.norm(x)
            return x # 输出 shape: (batch_size, num_patches, channels)

    return SwinFeatureExtractor(model)

MODEL_MAP = {
    'Pure_Resnet50':      (models.resnet50, get_resnet50_features),
    'vgg':         (models.vgg19, get_vgg19_features),
    'inception-v3':   (models.inception_v3, get_inceptionv3_features),
    'mobilenet-v2':   (models.mobilenet_v2, get_mobilenetv2_features),
    'convnext': (models.convnext_base, get_convnext_base_features),
    'swin-transformer':        (models.swin_b, get_swin_b_features),
}

class PerceptualLoss(nn.Module):
    def __init__(self, model_name: str, loss_fn: nn.Module = nn.L1Loss(), normalize_inputs: bool = True):
        super().__init__()

        if model_name not in MODEL_MAP:
            raise ValueError(f"不支持的模型: {model_name}. "
                             f"可用模型: {list(MODEL_MAP.keys())}")

        self.model_name = model_name
        self.normalize_inputs = normalize_inputs


        # 加载预训练模型
        model_loader, feature_hook = MODEL_MAP[model_name]
        pretrained_model = model_loader(pretrained=True)

        # 构建特征提取器
        self.feature_extractor = FeatureExtractor(pretrained_model, feature_hook)

        # 设置为评估模式，并冻结参数
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.loss_fn = nn.MSELoss(reduction = 'sum')

    def forward(self, generated_img, target_img):
        gen_features = self.feature_extractor(generated_img)
        target_features = self.feature_extractor(target_img)

        # 计算损失
        loss = self.loss_fn(gen_features, target_features)

        return loss


import torch.nn.functional as F

def image_complexity_metric(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    计算图像的复杂度度量。

    该函数使用Sobel算子计算图像的梯度幅度总和作为其复杂度的代理。
    复杂度越高的图像（如纹理丰富），其度量值也越高。

    Args:
        image_tensor (torch.Tensor): 输入的图像张量，形状为 (B, C, H, W)，
                                     数值范围建议为 [0, 1] 或 [-1, 1]。

    Returns:
        torch.Tensor: 一个形状为 (B,) 的张量，其中每个元素代表批次中
                      对应图像的复杂度得分。
    """
    if image_tensor.dim() != 4:
        raise ValueError("The input tensor must be 4 dimensional (B, C, H, W)")

    grayscale_tensor = 0.299 * image_tensor[:, 0:1, :, :] + \
                       0.587 * image_tensor[:, 1:2, :, :] + \
                       0.114 * image_tensor[:, 2:3, :, :]

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=image_tensor.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=image_tensor.device).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(grayscale_tensor, sobel_x, padding='same')
    grad_y = F.conv2d(grayscale_tensor, sobel_y, padding='same')

    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    complexity_score = torch.sum(grad_magnitude, dim=[1, 2, 3]) / (image_tensor.shape[2] * image_tensor.shape[3])
    # complexity_score = torch.sum(grad_magnitude, dim=[1, 2, 3])
    return complexity_score


def get_adaptive_t_star(
    x_ref: torch.Tensor,
    target_model,
    origin_label,
    target_model_name,
    m_avg,
    t_min: int = 18,
    t_max: int = 20,
    c_min: int = 40,
    c_max: int = 48,
    tao: float = 0.4,
) -> torch.Tensor:

    c_max_dict = {'mobilenet-v2': 60, 'vgg': 100, 'swin-transformer': 38, 'cubResnet50': 120, 'cubSEResnet154': 100,'cubSEResnet101': 170, 'carResnet50': 200, 'carSEResnet101': 200, 'carSEResnet154': 85}
    if 'cub' in target_model_name:
        t_min = 15
        c_min = 60
    if target_model_name=='swin-transformer':
        c_min = 30
        t_min = 19
    elif target_model_name=='cubSEResnet101':
        c_min = 90
    if target_model_name in c_max_dict.keys():
        c_max = c_max_dict[target_model_name]
    with torch.no_grad():

        out_image = (x_ref / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)
        outputs = target_model(out_image)
        probabilities = F.softmax(outputs, dim=1)
        prob = probabilities[0,origin_label].detach().cpu()

        x_ref = x_ref.cpu()
        complexity = image_complexity_metric(x_ref)

        adaptive_t = t_min + (t_max-t_min) / (1 + np.exp(-(complexity - m_avg) / tao))

        adaptive_const = c_min + (c_max-c_min) / (1 + np.exp(-(complexity - m_avg) / tao))*prob

        adaptive_t = torch.round(adaptive_t)

    return adaptive_t.to(torch.int), adaptive_const.to(torch.float32).cuda()

def comupte_average_complexity(config):

    complexities=[]
    if 'compatible' in config.input_image:
        gt_pth = './imagenet-compatible/images'
    elif 'CUB' in config.input_image:
        gt_pth = './CUB_200_2011/images'
    elif 'Car' in config.input_image:
        gt_pth = './Stanford_Car/images'

    gt_img_names = os.listdir(gt_pth)
    for gt_img_name in gt_img_names:
        gt_img_path = os.path.join(gt_pth, gt_img_name)
        image = normalize(Image.open(gt_img_path).convert("RGB"))
        complexity = image_complexity_metric(image).numpy()
        complexities.append(complexity)

    complex = torch.tensor(np.mean(complexities))

    return complex

import random

def random_color_shift(
    image_tensor: torch.Tensor,
    target_color_rgb,
    device: torch.device = None
):
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor. Got {type(image_tensor)}")
    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError(f"Input tensor must be of shape [B, 3, H, W]. Got {image_tensor.shape}")
    if not (-1.0 <= image_tensor.min() and image_tensor.max() <= 1.0):
        print("Warning: Input tensor values are outside [-1, 1] range. Clamping might occur.")
    img = image_tensor.clone()

    color_map = {
        "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
        "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
        "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128), "light green": (175, 238, 238),
        "light yellow": (175, 238, 238), "light blue": (100,197,255),
    }

    if isinstance(target_color_rgb, str):
        if target_color_rgb.lower() not in color_map:
            raise ValueError(f"can not recognize color name '{target_color_rgb}'. Please use {list(color_map.keys())} or offer RGB tuples。")
        rgb_tuple = color_map[target_color_rgb.lower()]
    else:
        rgb_tuple = target_color_rgb

    target_color_rgb_tensor = torch.tensor(rgb_tuple, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    target_color_rgb_0_to_1 = target_color_rgb_tensor / 255.0
    target_color_lab = kornia.color.rgb_to_lab(target_color_rgb_0_to_1)

    target_a = target_color_lab[0, 1, 0, 0]
    target_b = target_color_lab[0, 2, 0, 0]

    img_0_1 = (img + 1.0) / 2.0
    source_lab = kornia.color.rgb_to_lab(img_0_1)
    source_l = source_lab[:, 0:1, :, :]
    H, W = image_tensor.shape[2:]
    new_a = torch.full((1, 1, H, W), fill_value=target_a, device=device)
    new_b = torch.full((1, 1, H, W), fill_value=target_b, device=device)

    new_lab_image = torch.cat([source_l, new_a, new_b], dim=1)
    new_rgb_image_0_to_1 = kornia.color.lab_to_rgb(new_lab_image)
    output_tensor = new_rgb_image_0_to_1 * 2.0 - 1.0

    # random_number = random.random()
    # numbers = list(range(0,3))
    # chs = random.sample(numbers,2)
    # channel_changed = image_tensor.clone()
    # channel_changed[:,chs[0],:,:] = image_tensor[:,chs[1],:,:]

    return torch.clamp(output_tensor, -1.0, 1.0)

def get_entries(_pred_x0,target_model,target_label):

    out_image = (_pred_x0 / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)
    outputs = target_model(out_image)
    one_hot_labels = torch.eye(len(outputs[0]))[target_label].to(_pred_x0.device)
    i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())

    return i, j

