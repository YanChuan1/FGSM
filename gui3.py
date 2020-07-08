from tkinter import *
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image,ImageTk
import numpy as np
import cv2
import argparse
from net1 import CNN1
import os

model1 = CNN1(1, 10)
model2 = CNN1(1, 10)

saved1 = torch.load('./mnistpkl/relu.pkl', map_location='cpu')
saved2 = torch.load('./mnistpkl/advrelu.pkl', map_location='cpu')

model1.load_state_dict(saved1)
model2.load_state_dict(saved2)

model1.eval()
model2.eval()

path = './images'

i = 0
j = 0
for p in os.listdir(path):
	img = os.path.join(path, p)
	orig = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	img = orig.copy().astype(np.float32)

	IMG_SIZE = 28

	mean = [0.5]
	std = [0.5]
	img /= 255.0
	img = (img - mean)/std

	criterion = nn.CrossEntropyLoss()
	device = 'cpu'
	# prediction before attack
	inp1 = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
	inp2 = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
	
	out1 = model1(inp1)
	out2 = model2(inp2)

	pred1 = np.argmax(out1.data.cpu().numpy())
	pred2 = np.argmax(out2.data.cpu().numpy())

	loss1 = criterion(out1, Variable(torch.Tensor([float(pred1)]).to(device).long()))
	loss2 = criterion(out2, Variable(torch.Tensor([float(pred2)]).to(device).long()))

	loss1.backward()
	loss2.backward()

	inp1.data = inp1.data + ((80/255.0) * torch.sign(inp1.grad.data))
	inp1.data = inp1.data.clamp(min=-1, max=1)
	inp1.grad.data.zero_() 

	inp2.data = inp2.data + ((80/255.0) * torch.sign(inp2.grad.data))
	inp2.data = inp2.data.clamp(min=-1, max=1)
	inp2.grad.data.zero_()

	pred_adv1 = np.argmax(model1(inp1).data.cpu().numpy())
	pred_adv2 = np.argmax(model2(inp2).data.cpu().numpy())

	if int(pred1) == int(pred_adv1):
		i = i + 1

	if int(pred2) == int(pred_adv2):
		j = j + 1
	print("未使用对抗训练的模型，在被攻击前识别图像为：" + str(pred1))
	print("未使用对抗训练的模型，在被攻击后识别图像为：" + str(pred_adv1))
	print("使用对抗训练的模式，在被攻击前识别图像为：" + str(pred2))
	print("使用对抗训练的模型，在被攻击后识别图像为" + str(pred_adv2))

i = i * 100 / 20
j = j * 100 / 20
print("未使用对抗训练的模型，能抵御对抗攻击的概率是：" + str(i) + '%')
print("使用对抗训练的模型，能抵御对抗攻击的概率是：" + str(j) + '%')