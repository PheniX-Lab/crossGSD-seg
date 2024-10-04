import os

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from matplotlib import pyplot as plt
import mmcv
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm


config_file = r"/home/gaoy/SegFormer/local_configs/segformer/fade/segformer.b3_fade.512x512.ade.160k.py"
checkpoint_file = r"/home/gaoy/SegFormer/work_dirs/segformer.b3_fade.512x512.ade.160k/iter_16000.pth"
# checkpoint_file = r"F:\002Segformer\SegFormer-master\tools\output_weights\02wheat\wheat_sim2real.pth"


model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


#### format  must be .png  ####
img_root = r"/home/gaoy/00dataset/7suzhou/"
save_mask_root = r"/home/gaoy/00dataset/7suzhou_b3/"

if not os.path.exists(save_mask_root):
    os.mkdir(save_mask_root)
img_names = os.listdir(img_root)
for img_name in tqdm(img_names):
    # test a single image
    img = img_root + img_name
    result = inference_segmentor(model, img)[0]
    img = Image.fromarray(np.uint8(result*255))
    img.save(save_mask_root + img_name)
