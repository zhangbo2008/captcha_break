#可算搞定了!!!!!!!!!!!!!!


from captcha.image import ImageCaptcha
from CaptchaDataset import CaptchaDataset
import string
characters = '-' + string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 192, 64, 4, len(characters)#192 64
n_input_length = 12
import random
label_length=n_len



random_str = ''.join([random.choice(characters[1:]) for j in range(label_length)])
print("随机生成的图片是test.png","文字是",random_str)
image = ImageCaptcha(width=width, height=height).write(random_str,output='./test.png', format='png')


#下面开始识别.

import numpy as np
from Model import Model
data=np.array('./test.png')
print(data)


import torch
import matplotlib.image as img
image = img.imread('./test.png')
image= torch.tensor(image).permute(2,0,1)


import torch
import torch.nn as nn
import torch.nn.functional as F


model=torch.load('ctc3.pth')
model.eval()
output =model(image.unsqueeze(0).cuda())
print(output)
print(output.shape)


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s



output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
print('pred:', decode(output_argmax[0]))
print('over')
