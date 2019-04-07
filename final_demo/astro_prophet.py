from __future__ import division
from model import ft_net
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import pandas as pd
import random
import argparse
import pickle as pkl
import os
import shutil
import re
from PIL import Image
from torchvision import datasets, transforms, models
data_dir='/home/luke/'


#transform_val_list = [
#			transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
#			transforms.ToTensor(),
#			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#			]
#
test_transforms = transforms.Compose([transforms.Resize(size=(256,128),interpolation=3),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
device=torch.device('cpu')
imsize = 256
model1=ft_net(6)
#model.load_state_dict(torch.load(file_path))
model1.load_state_dict(torch.load('net_9.pth'))
model1.eval()
def predict_image(image_p):
	image=Image.open(image_p)
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	input = input.to(device)
	print (input.shape)
	output = model1(input)
	index = output.data.cpu().numpy().argmax()
	return index


print(predict_image('2.png'))
