import sys
import os 
import random
import argparse

sys.path.append(".")
sys.path.append('./taming-transformers')

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim_adv import DDIMSampler
from torchvision import models

from torch.backends import cudnn

import numpy as np
from torchvision.transforms import ToPILImage

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--ddim-steps', type=int, default=200)
parser.add_argument('--ddim-eta', type=float, default=0.0)
parser.add_argument('--mask', type=bool, default=True)
parser.add_argument('--save-dir', type=str, default='../images/imagenetsc/')
parser.add_argument('--target-model', type=str, default="inception-v3")
parser.add_argument('--adaptive', type=bool, default=True)
parser.add_argument('--const', type=float, default=-30)
parser.add_argument('--lr', type=float, default=0.03)
parser.add_argument('--iterations', type=int, default=2)
args = parser.parse_args()

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def normalize(image, shape=(256, 256)):
    image = np.array(image.convert("RGB").resize(shape))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image * 2.0 - 1.0
    return image

def prepare_target_model(target_model, device):
    if target_model=='Pure_Resnet50':
        print("loading Pure Resnet50")
        model = models.resnet50(pretrained=True).to(device).eval()
    elif target_model == 'mobilenet-v2':
        print("loading mobilenet")
        model = models.mobilenet_v2(pretrained=True).to(device).eval()
    elif target_model == 'inception-v3':
        print("loading inception")
        model = models.inception_v3(pretrained=True).to(device).eval()
    elif target_model == 'swin-transformer':
        print("loading swin-transformer")
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        model.to(device).eval()
    elif target_model == 'vgg':
        print("loading vgg")
        model = models.vgg19(pretrained=True)
        model.to(device).eval()
    else:
        raise NotImplementedError("No such model!")
    return model

def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model


def main():
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    outpath = os.path.join(args.save_dir, args.target_model)+'/samples'
    os.makedirs(outpath, exist_ok=True)
    
    model = get_model()
    vic_model = prepare_target_model(args.target_model, device)
    sampler = DDIMSampler(model, args, vic_model=vic_model)
    n_samples_per_class = args.batch_size

    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale   # for unconditional guidance

    classes =  np.arange(1000)
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )

            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label], dtype=torch.long)
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 label=xc.to(model.device),
                                                 args=args,
                                                 )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                             min=0.0, max=1.0)
                for i in range(args.batch_size):
                    img_name = str(class_label)+'_'+str(i)+'.png'
                    pil = ToPILImage()(x_samples_ddim[i,:,:,:])
                    pil.save(os.path.join(outpath, img_name))

            
if __name__ == '__main__':
    main()
