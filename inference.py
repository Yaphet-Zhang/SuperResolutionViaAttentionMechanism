# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:10:41 2021

@author: zhangbowen
"""

import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SRCNN, AttentionZBW

def calc_psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    return 10*np.log10(255*255/mse)

#parser = argparse.ArgumentParser(description='SRCNN run parameters')
#parser.add_argument('--model', type=str, required=True)
#parser.add_argument('--image', type=str, required=True)
#parser.add_argument('--zoom_factor', type=int, required=True)
#parser.add_argument('--cuda', action='store_true')
#args = parser.parse_args()

# parameters
model_name='srcnn'
# model_name='attentionzbw'
img_path = r'3.jpg'
scale = 4

# check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device：", device)
print('----------')

##### load SRCNN & AttentionZBW #####
if model_name == 'srcnn':
    model = SRCNN()
    model.to(device)
    print("Model：", model)
    print('----------')
    weights_path = 'weights/srcnn/srcnn_1000.pth'
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
elif model_name == 'attentionzbw':
    model = AttentionZBW()
    model.to(device)
    print("Model：", model)
    print('----------')
    weights_path = 'weights/attentionzbw/attentionzbw_1000.pth'
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

##### write BICUBIC #####
img = Image.open(img_path).convert('RGB')
img_bicubic = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.BICUBIC)
img_bicubic.save(img_path.replace('.', '_bicubic_x%d.' % scale))

##### preprocessing #####
# open original image
img = Image.open(img_path).convert('YCbCr')
print('Original iamge size[w,h]:', img.size)
print('----------')

# upscale the image (bicubic)
img_bicubic = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.BICUBIC)
y_channel, cb, cr = img_bicubic.split()
print('Bicubic iamge size[w,h]:', img_bicubic.size)
print('----------')

##### evaluate bicubic (PSNR) #####
eval_img = img
eval_img_y, _, _ = eval_img.split()
eval_bicubic = img_bicubic.resize((int(img.size[0]), int(img.size[1])), Image.BICUBIC)
eval_bicubic_y, _, _ = eval_bicubic.split()
psnr_bicubic = calc_psnr(np.array(eval_img_y), np.array(eval_bicubic_y))
print('PSNR of img&bicubic: %.3f' % psnr_bicubic)
print('----------')

##### inference #####
# ([W,H]-->[N,C,H,W] & [0,255]-->[0,1])
img_to_tensor = transforms.ToTensor()
# only Y channel is used
input_tensor = img_to_tensor(y_channel).view(1, -1, y_channel.size[1], y_channel.size[0])

input_tensor = input_tensor.to(device)
print('input:', input_tensor.shape)
print('----------')
output_tensor = model(input_tensor).cpu()
print('output:', output_tensor.shape)
print('----------')

# [C,N,H,W]-->[C,H,W]) & [0,1]-->[0,255]
output_numpy = output_tensor[0].detach().numpy()
output_numpy *= 255.0

# convert output_numpy from [-xx,+xx] to [0,255]
output_numpy = output_numpy.clip(0, 255)

# convert numpy to PIL.Image.Image & convert float32 to uint8
output_img = Image.fromarray(np.uint8(output_numpy[0]), mode='L')

##### evaluate SRCNN (PSNR) & AttentionZBW (PSNR) #####
if model_name == 'srcnn':
    eval_srcnn_y = output_img.resize((int(img.size[0]), int(img.size[1])), Image.BICUBIC)
    psnr_srcnn = calc_psnr(np.array(eval_img_y), np.array(eval_srcnn_y))
    print('PSNR of img&srcnn: %.3f' % psnr_srcnn)
    print('----------')
elif model_name == 'attentionzbw':
    eval_attentionzbw_y=output_img.resize((int(img.size[0]), int(img.size[1])), Image.BICUBIC)
    psnr_attentionzbw=calc_psnr(np.array(eval_img_y),np.array(eval_attentionzbw_y))
    print('PSNR of img&attentionzbw: %.3f'%psnr_attentionzbw)
    print('----------')

# merge Y, Cb, Cr   &  convert ycbcr to RGB  
output_rgb = Image.merge('YCbCr', [output_img, cb, cr]).convert('RGB')

##### write SRCNN & AttentionZBW image #####
if model_name == 'srcnn':
    output_rgb.save(img_path.replace('.', '_srcnn_x%d.' % scale))
elif model_name == 'attentionzbw':
    output_rgb.save(img_path.replace('.', '_attentionzbw_x%d.' % scale))
