"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torchvision.transforms as T

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from torchvision.models import resnet50
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

#weights = ResNet50_Weights.DEFAULT
#preprocess = weights.transforms()

preprocess = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_target_label(logits, label, device): # seond-like label for attack
    
    rates, indices = logits.sort(1, descending=True) 
    rates, indices = rates.squeeze(0), indices.squeeze(0)
    tar_label = torch.zeros_like(label).to(device)
    if label == indices[0]:
        tar_label = indices[1]
    else:
        tar_label = indices[0]
    # for i in range(label.shape[0]):
    #     if label[i] == indices[i][0]:  # classify is correct
    #         tar_label[i] = indices[i][1]
    #     else:
    #         tar_label[i] = indices[i][0]
    
    return tar_label


class DDIMSampler(object):
    def __init__(self, model, args, schedule="linear", vic_model=None, **kwargs):
        
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.vic_model = vic_model
        self.args = args
        self.reg_coef_recon = self.args.target_model != 'inception-v3'


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               label=None,reference=None, K=10,s=2,a=1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,label=label, reference=reference,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,label=None,reference=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]


        pri_img = img.detach().requires_grad_(True)
        img_origin = img

        img = pri_img.detach().requires_grad_(True)

        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # print(time_range[-40],img.shape)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,img_origin=img_origin, label=label,reference=reference)
            img, pred_x0, img_origin = outs


            '''
            if index % 20 == 0:
                x_samples_ddim = self.model.decode_first_stage(img)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                save_image(x_samples_ddim, f"img/Diff_{index}.png", nrow=1, normalize=True)
            '''

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

                
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,img_origin=None, label=None, reference=None):
        b, *_, device = *x.shape, x.device
        total_steps = self.ddim_timesteps.shape[0]
        const = self.args.const
        lr_xt = self.args.lr
        iterations = self.args.iterations

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            if reference is None and total_steps * 0 < index <= total_steps * 0.2:
                e_t_origin = self.model.apply_model(img_origin, t, c)
            else:
                e_t_origin = e_t
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            if reference is None and total_steps * 0 <= index <= total_steps * 0.2:
                x_in_origin = torch.cat([img_origin] * 2)
                e_t_uncond_origin, e_t_origin = self.model.apply_model(x_in_origin, t_in, c_in).chunk(2)
                e_t_origin = e_t_uncond_origin + unconditional_guidance_scale * (e_t_origin - e_t_uncond_origin)
            else:
                e_t_origin = e_t

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            e_t_origin= score_corrector.modify_score(self.model, e_t_origin, img_origin, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if reference is None and total_steps * 0 <= index <= total_steps * 0.2:
            origin_pred_x0 = (img_origin - sqrt_one_minus_at * e_t_origin) / a_t.sqrt()
        else:
            origin_pred_x0 = pred_x0


        if (index >= total_steps * 0 and index <= total_steps * 0.2):
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            with torch.enable_grad():
                origin_x = x.clone().detach()
                x = x.detach().requires_grad_(True)

                img_transformed_origin = self.model.differentiable_decode_first_stage(
                    origin_pred_x0)  # image transformation from latent code
                img_transformed_origin = torch.clamp((img_transformed_origin + 1.0) / 2.0,
                                                     min=0.0, max=1.0)
                img_transformed_origin = preprocess(img_transformed_origin).to(
                    device)  # image transformation to model input
                logits_origin = self.vic_model(img_transformed_origin)
                probs = F.softmax(logits_origin)
                prob = probs[0, label]
                if self.args.adaptive:
                    const = const - 10 * prob

                for step in range(iterations):
                    img_transformed = self.model.differentiable_decode_first_stage(
                        pred_x0)  # image transformation from latent code
                    img_transformed = torch.clamp((img_transformed + 1.0) / 2.0,
                                                  min=0.0, max=1.0)
                    img_transformed = preprocess(img_transformed).to(device)  # image transformation to model input


                    logits = self.vic_model(img_transformed)
                    temp = torch.eye(len(logits[0])).cuda()
                    one_hot_labels = temp[label].to(pred_x0.device)
                    i, _ = torch.max((1 - one_hot_labels) * logits, dim=1)
                    j = torch.masked_select(logits, one_hot_labels.bool())
                    ret = (const - (j - i)) ** 2 + self.reg_coef_recon*torch.sum((origin_pred_x0 - pred_x0) ** 2)
                    loss = ret+0.01*torch.sum((origin_x - x) ** 2)

                    x_grad = torch.autograd.grad(
                        loss, x, retain_graph=False, create_graph=False
                    )[0].detach()
                    print(torch.norm(x_grad, p=2))
                    if torch.norm(x_grad, p=2).isnan():
                        new_x = x
                    else:
                        new_x = x - lr_xt * x_grad
                    x = new_x.detach().requires_grad_(True)
                    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                        e_t = self.model.apply_model(x, t, c)
                    else:
                        x_in = torch.cat([x] * 2)
                        t_in = torch.cat([t] * 2)
                        c_in = torch.cat([unconditional_conditioning, c])
                        e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    del loss, x_grad
                    torch.cuda.empty_cache()

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            if reference is None and total_steps * 0 <= index <= total_steps * 0.2:
                origin_pred_x0, _, *_ = self.model.first_stage_model.quantize(origin_pred_x0)
            else:
                origin_pred_x0 = pred_x0
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        if reference is None and total_steps * 0 <= index <= total_steps * 0.2:
            dir_xt_origin = (1. - a_prev - sigma_t**2).sqrt() * e_t_origin
        else:
            dir_xt_origin = dir_xt
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        img_origin = a_prev.sqrt() * origin_pred_x0 + dir_xt_origin + noise
        return x_prev, pred_x0, img_origin

