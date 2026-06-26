# A diffusion Prior-based Framework for Imperceptible and Flexible Unrestricted Adversarial Attacks

This is the official impelmenation of the paper [A diffusion Prior-based Framework for Imperceptible and Flexible Unrestricted Adversarial
Attacks].

## Abstract
Unrestricted Adversarial Examples (UAEs) pose a growing security challenge to Deep Neural Networks by introducing 
substantial, semantically natural modifications to images. While current diffusion-based methods improve 
the naturalness of UAEs, they suffer from two key limitations: an underutilization of the diffusion model's 
learned data distribution, which caps sample quality, and a lack of flexibility for diverse attack scenarios. 
To address these issues, we first reframe the UAE generation as a Bayesian inference problem, leveraging a 
pre-trained diffusion model as a powerful prior to ensure UAEs are statistically consistent with real data 
and therefore improve their effectiveness and naturalness. Building upon this foundation, we introduce DiffAdvMAP+, 
an enhanced framework featuring a novel sample-specific adaptive mechanism. This module dynamically 
tunes key parameters, such as the diffusion step and adversarial confidence level, based on individual 
image complexity, thereby boosting both efficacy and efficiency. This principled approach, 
combining a flexible reconstruction constraint with sample-specific adaptation, 
allows DiffAdvMAP+ to excel at diverse tasks, including noise-originating and image-similar attacks. 
Extensive experiments conducted on various model structures, datasets, and defense methods demonstrate 
that our method achieves a superior trade-off between image quality, flexibility, and transferability 
compared to existing approaches. 

## Requirements
### Environment
Install conda environment by running
```
conda env create -f environment.yaml
conda activate DiffAdvMAP
```
Download ImageNet checkpoint: [guided-diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and save to "checkpoints/256x256_diffusion_uncond.pt“, [ldm](https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt) and save to "from_scratch/models/ldm/cin256-v2/model.ckpt"

Then download [Imagenet-Compatible](https://drive.google.com/file/d/1sAD1aVLUsgao1X-mu6PwcBL8s68dm5U9/view?usp=sharing) and change the settings of `input_image` in config file `imagenet_perturb.yaml`.

## Usage
### Generating UAEs
To generate UAEs with reference images, you can run 
```text
python main.py:
    --config_file:  The configuration file, which specifies the model to use and some hyper-parameters for our method
    --input_image:  The path to input image
    --outdir:       The path to output folder
    --adaptive:     Using adaptive adversarial confidence level and diffusion step or not
    --const:        If adaptive is False, generate UAEs with the const adversarial confidence level (it should be no larger than 0)
    --target_model: The surrogate model to be attacked
    --regional:     Generating regional UAEs or not
    --mask:         The path to mask file
    --style:        The path to style image for generating style UAEs
    --reference_color_name: The reference color for generating color UAEs
```
Here is an example result on generating global image-similar with inception-v3:
```shell
python main.py --config_file configs/imagenet_perturb.yaml --input_image ./imagenet-compatible/images --target_model inception-v3 --adaptive True --regional False --outdir ./images/imagenet/ 
```
Here is an example result on generating regional color UAEs with inception-v3:
```shell
python main.py --config_file configs/imagenet_color.yaml --input_image ./imagenet-compatible/images --masks ./maks --reference_color_name blue --target_model inception-v3 --adaptive False --regional True --outdir ./images/imagenetcl/
```
To generate UAEs from noise, you can run
```shell
cd from_scratch
python from_scratch.py --target_model inception-v3 
```
The generated UAEs will be saved in direction: `./<outdir>/<target_model>/samples/`
### Tip:
You can tune following hyper-parameters if you are not satisfied with the generated image:
* `--optimize_xt.num_iteration_optimize_xt`: The number of optimization steps $G$.
* `--optimize_xt.lr_xt`: The initial learning rate $\mu_T$


Please refer to our paper for more details.

## Attack CUB_200_2011 and Standford Cars datasets
Follow [DiffAttack](https://github.com/WindVChen/DiffAttack), we use the same subset from `CUB_200_2011` and `Stanford Cars` datasets. You can download the 
dataset here [[CUB_200_2011](https://drive.google.com/file/d/1umBxwhRz6PIG6cli40Fc0pAFl2DFu9WQ/view?usp=sharing)|[Stanford Cars](https://drive.google.com/file/d/1FiH98QyyM9YQ70PPJD4-CqOBZAIMlWJL/view?usp=sharing)]
You should also download pre-trained models of CUB_200_2011 and Stanford Cars form [Beyond-ImageNet-Attack](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack)
Put them into the `pretrianed_models` folder, then run our attack method:
```shell
python main.py --config_file configs/cub_perturb.yaml --input_image ./CUB_200_2011/images --target_model cubSEResnet101 --adaptive True --regional False --outdir ./images/cub/ 
```
## Evaluation
To evaluate the generated UAEs on all target models in our paper, run:
```shell
python test_classifier.py --config_file <config_file> --use_specific_path <True or False> --specific_path <your path of UAEs> --adversarial_trained <True of False>
```
If `use_pecific_path` is `False`, UAEs in config file's default image folder will be evaluated
## Robustness on defensive approches
Apart from the adversarially trained models, we also evaluate our attack's power to deceive other defensive approaches as displayed in Section 4.C.2) in our paper, their implementations are as follows:
- Adversarially trained models (Adv-Inc-v3, Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub>): Run the code in [Evaluation](#Evaluation), you can download the pretrained weights from [here](https://github.com/ylhz/tf_to_pytorch_model) and then place them into the directory `pretrained_models`.
- [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense): Since our target size is 224, we reset the image scale augmentation proportionally (232~248). Then run the original code.
- [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): Since its ensembled models failed to process inputs with 224 size, we run its original code that resized the inputs to 299 size.
- [SR](https://github.com/aamir-mustafa/super-resolution-adversarial-defense): Change the input size to 224 then run the original code.
- [NRP](https://github.com/Muzammal-Naseer/NRP): Change the input size to 224 and set purifier=NRP, dynamic=True, then run the original code.
- [DiffPure](https://github.com/NVlabs/DiffPure): Modify the original codes to evaluate the existing adversarial examples, not crafted examples again.


Our implementation is based on following repos:
* https://github.com/WindVChen/DiffAttack
* https://github.com/openai/guided-diffusion
* https://github.com/CompVis/latent-diffusion
* https://github.com/UCSB-NLP-Chang/CoPaint
