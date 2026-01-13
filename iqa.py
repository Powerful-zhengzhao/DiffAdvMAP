import json
import os
import torch
from PIL import Image
import numpy as np
import os
import pyiqa

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def normalize(image, shape=(256, 256)):

    image = np.array(image.convert("RGB").resize(shape))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image)
    image = image[None].permute(0, 3, 1, 2)
    return image

hyper_metric = pyiqa.create_metric('hyperiqa', device=device)
tres_metric = pyiqa.create_metric('tres', device=device)

path = './images/imagenetsc/mobilenet-v2/samples'
images = os.listdir(path)
hyper = 0
tres = 0
for image in images:
    image_path = os.path.join(path, image)
    im = Image.open(image_path)
    img = normalize(im).cuda()
    score_hp = hyper_metric(img)
    score_tr = tres_metric(img)
    hyper += score_hp
    tres += score_tr
print('the hyperiqa score is {},the tres score is {}'.format( hyper/len(images),tres/len(images)))

