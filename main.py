import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from datasets.utils import normalize
from guided_diffusion import (
    DDIMSampler,
    DDNMSampler,
    DDRMSampler,
    DPSSampler,
    A_DDIMSampler,
)
from guided_diffusion import dist_util
from guided_diffusion.ddim import R_DDIMSampler
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
)
from metrics import  LPIPS, PSNR, SSIM, Metric
from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer

import torchvision.models as models
import torch.nn as nn
import json
import random
from guided_diffusion.attacktool import style_transfer
import torch.nn.functional as F
from Finegrained_model import model as otherModel
from guided_diffusion.attacktool import comupte_average_complexity, get_adaptive_t_star, random_color_shift
from torchvision import transforms
from collections import defaultdict

import os
from guided_diffusion.evaluation import evaluation



def prepare_model(algorithm, conf, device):
    logging_info("Prepare model...")
    unet = create_model(**select_args(conf, model_defaults().keys()), conf=conf)
    SAMPLER_CLS = {
        "repaint": SpacedDiffusion,
        "ddim": DDIMSampler,
        "resample": R_DDIMSampler,
        "ddnm": DDNMSampler,
        "ddrm": DDRMSampler,
        "dps": DPSSampler,
        "a_ddim": A_DDIMSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )
    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu"
        ), strict=False
    )
    unet.to(device)
    if conf.use_fp16:
        unet.convert_to_fp16()
    unet.eval()
    return unet, sampler

def prepare_data(config,target_model):
    index_path = json.load(open(config.index_path))
    if config.resume==True:
        save_path = os.path.join(config.outdir,config.target_model, 'samples')
        finished = os.listdir(save_path)
    else:
        finished = []
    num = len(finished)
    finished_imgs=[]
    for i in range(num):
        f = str(i+1)+'.png'
        finished_imgs.append(f)
    datas = []
    gt_pth = config.input_image
    masks_path = config.mask
    label_dict = defaultdict(int)

    if os.path.isdir(gt_pth):
        if 'imagenet' in config.input_image:
            label_path = './imagenet-compatible/labels.txt'
        elif 'CUB' in config.input_image:
            label_path = './CUB_200_2011/labels.txt'
        elif 'Car' in config.input_image:
            label_path = './Stanford_Car/labels.txt'

        with open(label_path, 'r') as file:
            content = file.read()
            lines = content.split('\n')
            for i, line in enumerate(lines):
                image_name = str(i + 1) + '.png'
                origin_label = int(line.strip()) - 1
                label_dict[image_name] = origin_label
            file.close()
        m_avg = comupte_average_complexity(config)

        for key,value in label_dict.items():
            image_name = key
            origin_label = value
            image_path = os.path.join(gt_pth, image_name)
            image = normalize(Image.open(image_path).convert("RGB"))
            if config.mode == "perturbation_attack":
                mask_name = image_name.split('.')[0] + '.png'
                mask_path = os.path.join(masks_path, mask_name)
                m = Image.open(mask_path).convert("1")
                mask = (
                    torch.from_numpy(1-np.array(m, dtype=np.float32))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                # ratio = 0.4
                # b, c, h, w = image.shape
                # mask = torch.ones_like(image)
                # m_h, m_w = int(h * ratio), int(w * ratio)
                # for b in range(b):
                #     # mask mask[b]
                #     x_s = random.randint(0, h - m_h)
                #     y_s = random.randint(0, w - m_w)
                #     mask[b][:, x_s:x_s + m_h, y_s:y_s + m_w] = 0
                #     pil = mask.clone()
                #     pil = 1-pil
                #     pil = ToPILImage()(pil.squeeze(0).cpu().data)
                #     pil.save(mask_path)
            elif config.mode == "style" or config.mode == "colorization":
                mask_name = image_name.split('.')[0] + '.png'
                mask_path = os.path.join(masks_path, mask_name)
                m = Image.open(mask_path).convert("1").resize((224, 224))
                mask = (
                    torch.from_numpy(1 - np.array(m, dtype=np.float32))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            t_star, adaptive_const = get_adaptive_t_star(image.cuda(), target_model=target_model,
                                                         origin_label=origin_label,
                                                         target_model_name=config.target_model,
                                                         m_avg=m_avg)
            if image_name not in finished_imgs:
                datas.append((image, mask, index_path[str(origin_label)][0], origin_label, origin_label,t_star,adaptive_const))
    else:
        image = normalize(Image.open(gt_pth).convert("RGB"))

        out_image = (image / 2 + 0.5).clamp(0, 1).cuda()
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)
        outputs = target_model(out_image)
        origin_label = torch.argmax(outputs, dim=1)
        origin_label = origin_label[0].cpu().numpy()
        print(origin_label)
        if config.mode == "perturbation_attack":
            m = Image.open(masks_path).convert("1")
            mask = (
                torch.from_numpy(np.array(m, dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
        elif config.mode == "style" or config.mode == "colorization":
            m = Image.open(masks_path).convert("1").resize((224, 224))
            mask = (
                torch.from_numpy(1 - np.array(m, dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
        t_star = 15
        adaptive_const = config.const
        datas.append(
            (image, mask, index_path[str(origin_label)][0], origin_label, origin_label, t_star, adaptive_const))

    logging_info(f"Load {len(datas)} samples")

    return datas,num


def all_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            return False
    return True

def prepare_target_model(config, device):
    if config.target_model=='Pure_Resnet50':
        print("loading Pure Resnet50")
        model = models.resnet50(pretrained=True).to(device).eval()
    elif config.target_model == 'mobilenet-v2':
        print("loading mobilenet")
        model = models.mobilenet_v2(pretrained=True).to(device).eval()
    elif config.target_model == 'inception-v3':
        print("loading inception")
        model = models.inception_v3(pretrained=True).to(device).eval()
    elif config.target_model == 'swin-transformer':
        print("loading swin-transformer")
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        model.to(device).eval()
    elif config.target_model == 'vgg':
        print("loading vgg")
        model = models.vgg19(pretrained=True)
        model.to(device).eval()
    elif config.target_model == 'cubResnet50':
        print("loading cubResnet50")
        model = otherModel.CUB()[0]
        model = model.to(device).eval()
    elif config.target_model == 'cubSEResnet154':
        print("loading cubSEResnet154")
        model = otherModel.CUB()[1]
        model = model.to(device).eval()
    elif config.target_model == 'cubSEResnet101':
        print("loading cubSEResnet101")
        model = otherModel.CUB()[2]
        model = model.to(device).eval()
    elif config.target_model == 'carResnet50':
        print("loading carResnet50")
        model = otherModel.CAR()[0]
        model = model.to(device).eval()
    elif config.target_model == 'carSEResnet154':
        print("loading carSEResnet154")
        model = otherModel.CAR()[1]
        model = model.to(device).eval()
    elif config.target_model == 'carSEResnet101':
        print("loading carSEResnet101")
        model = otherModel.CAR()[2]
        model = model.to(device).eval()
    else:
        raise NotImplementedError("No such model!")
    return model

def main():
    ###################################################################################
    # prepare config, logger and recorder
    ###################################################################################
    config = Config(default_config_file="configs/imagenet_perturb.yaml", use_argparse=True)
    # config.show()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    all_paths = get_all_paths(config.outdir)
    config.dump(all_paths["path_config"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    recorder = ResultRecorder(
        path_record=all_paths["path_record"],
        initial_record=config,
        use_git=config.use_git,
    )
    set_random_seed(config.seed, deterministic=False, no_torch=False, no_tf=True)

    #get surrogate model
    target_model = prepare_target_model(config, device)
    print('target model loaded')
    ###################################################################################
    # prepare data
    ###################################################################################
    datas,num = prepare_data(config, target_model)

    ###################################################################################
    # prepare model and device
    ###################################################################################
    unet, sampler = prepare_model(config.algorithm, config, device)
    def model_fn(x, t, y=None, gt = None, **kwargs):
        return unet(x, t, None, gt=gt)
    cond_fn = None
    METRICS = {
        "lpips": Metric(LPIPS("alex", device)),
    }

    ###################################################################################
    # start sampling
    ###################################################################################
    logging_info("Start sampling")
    timer, num_image = Timer(), 0
    batch_size = config.n_samples
    i=num
    for data in tqdm(datas):
        image, mask, image_name, class_id, target_label,t_star,adaptive_const = data
        origin_label = class_id
        print('loading image {} for model {}'.format(image_name, config.target_model))
        # prepare save dir
        if config.regional == True:
            outpath = os.path.join(outpath, "regional")
        else:
            outpath = config.outdir
        outpath = os.path.join(outpath, config.target_model)
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        image = image.to(device)
        mask = mask.to(device)
        i += 1
        if config.mode == 'style':
            style_image = normalize(Image.open(config.style).convert("RGB"))
            style_image = style_image.to(device)
            # img = style_image
            img = style_transfer((image + 1.0) / 2.0, (style_image + 1.0) / 2.0, mask, content_w=1, style_w=4000,
                                 num_iters=1000)
            img = img * 2.0 - 1.0
            # img_name = os.path.basename(config.input_image).split(".")[0]+'_ref.jpg'
            # I = (img + 1.0) / 2.0
            # save_image(I[0], os.path.join(sample_path, img_name))
        elif config.mode == 'colorization':
            reference_color_name = config.reference_color_name
            img = random_color_shift(image,reference_color_name, device)
            # img_name = os.path.basename(config.input_image).split(".")[0]+'_ref.jpg'
            # I = (img + 1.0) / 2.0
            # save_image(I[0], os.path.join(sample_path, img_name))
        else:
            img = torch.zeros_like(image)

        # prepare batch data for processing
        batch = {"image": image.to(device), "mask": mask.to(device),"reference_image": img.to(device)}
        model_kwargs = {
            "gt": batch["image"].repeat(batch_size, 1, 1, 1),
            "gt_keep_mask": batch["mask"].repeat(batch_size, 1, 1, 1),
            "reference_image": batch["reference_image"].repeat(batch_size, 1, 1, 1),
        }

        shape = (batch_size, 3, 224, 224)

        # sample images
        samples = []
        for n in range(config.n_iter):
            timer.start()
            result = sampler.p_sample_loop(
                model_fn,
                shape=shape,
                adaptive_const=adaptive_const,
                t_star = t_star,
                target_model=target_model,
                target_label=target_label,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=True,
                return_all=True,
                conf=config,
                sample_dir=outpath if config["debug"] else None,
            )
            timer.end()

            for metric in METRICS.values():
                metric.update(result["sample"], batch["image"])

            adv = normalize_image(result["sample"])
            out_image = adv.permute(0, 2, 3, 1)
            mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
            out_image = out_image[:, :, :].sub(mean).div(std)
            out_image = out_image.permute(0, 3, 1, 2)
            outputs = target_model(out_image)
            label = torch.argmax(outputs, dim=1)
            label = label[0].cpu().numpy()
            logging_info(
                "the predictied label of the UAE is: %d, the origin label of the UAE is: %d"
                % (label, target_label)
            )
            samples.append(adv.detach().cpu())

        samples = torch.cat(samples)
        # save generations
        for sample in samples:

            if os.path.isdir(config.input_image):
                img_name = str(origin_label)+'_'+str(i)+'.png'
            else:
                img_name = os.path.basename(config.input_image)
            save_image(sample, os.path.join(sample_path, img_name))
            base_count += 1

        # save metrics
        for metric_name, metric in METRICS.items():
            torch.save(metric.dataset_scores[-config.n_iter:], os.path.join(outpath, metric_name + ".last"))

        num_image += 1
        last_duration = timer.get_last_duration()
        logging_info(
            "It takes %.3lf seconds for image %s"
            % (float(last_duration), image_name+str(i))
        )
    if os.path.isdir(config.input_image):
        evaluation(config)
    # report batch scores
    for metric_name, metric in METRICS.items():
        recorder.add_with_logging(
            key=f"{metric_name}_score_{image_name}_{i}",
            value=metric.report_batch(),
        )

    # report over all results
    for metric_name, metric in METRICS.items():
        mean, colbest_mean = metric.report_all()
        recorder.add_with_logging(key=f"mean_{metric_name}", value=mean)
        recorder.add_with_logging(
            key=f"best_mean_{metric_name}", value=colbest_mean)

    if num_image > 0:
        recorder.add_with_logging(
            key="mean time", value=timer.get_cumulative_duration() / num_image
        )

    logging_info(
        f"Your samples are ready and waiting for you here: \n{config.outdir} \n"
        f" \nEnjoy."
    )
    recorder.end_recording()


if __name__ == "__main__":
    main()
