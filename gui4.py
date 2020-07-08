import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image,ImageTk
import numpy as np
import cv2
from net1 import CNN1
from net2 import CNN2
from net3 import CNN3
import os

def fun(img):
    eps = 0
    i = 0
    flag1 = 0
    j = 0
    flag2 = 0
    k = 0
    flag3 = 0
    while True:
        eps = eps + 1 
        inp1 = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
        inp2 = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
        inp3 = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)

        out1 = model1(inp1)
        out2 = model2(inp2)
        out3 = model3(inp3)

        pred1 = np.argmax(out1.data.cpu().numpy())
        pred2 = np.argmax(out2.data.cpu().numpy())
        pred3 = np.argmax(out3.data.cpu().numpy())

       # print('Prediction before attack--relu:'+ str(pred1) +  '  leakyrelu:' + str(pred2) + '  softplus:' + str(pred3))

        loss1 = criterion(out1, Variable(torch.Tensor([float(pred1)]).to(device).long()))
        loss2 = criterion(out2, Variable(torch.Tensor([float(pred2)]).to(device).long()))
        loss3 = criterion(out3, Variable(torch.Tensor([float(pred3)]).to(device).long()))
        # compute gradients
        loss1.backward()
        loss2.backward()
        loss3.backward()
        # this is it, this is the method
        inp1.data = inp1.data + ((eps/255.0) * torch.sign(inp1.grad.data))
        inp1.data = inp1.data.clamp(min=-1, max=1)
        inp1.grad.data.zero_() 

        inp2.data = inp2.data + ((eps/255.0) * torch.sign(inp2.grad.data))
        inp2.data = inp2.data.clamp(min=-1, max=1)
        inp2.grad.data.zero_()

        inp3.data = inp3.data + ((eps/255.0) * torch.sign(inp3.grad.data))
        inp3.data = inp3.data.clamp(min=-1, max=1)
        inp3.grad.data.zero_() 

        # predict on the adversarial image
        pred_adv1 = np.argmax(model1(inp1).data.cpu().numpy())
        pred_adv2 = np.argmax(model2(inp2).data.cpu().numpy())
        pred_adv3 = np.argmax(model3(inp3).data.cpu().numpy())
       # print('Prediction after attack--relu:'+ str(pred_adv1) +  '  leakyrelu:' + str(pred_adv2) + '  softplus:' + str(pred_adv3))
        
        if pred1 != pred_adv1 and flag1 == 0:
            i = eps
            flag1 = 1

        if pred2 != pred_adv2 and flag2 == 0:
            j = eps
            flag2 = 1

        if pred3 != pred_adv3 and flag3 == 0:
            k = eps
            flag3 = 1

        if flag1 == 1 and flag2 == 1 and flag3 == 1:
            break

        if eps == 200:
            if(flag1 == 0):
                i = 128
            if(flag2 == 0):
                j = 128
            if(flag3 == 0):
                k = 128
            break

    print('i is %d' , i)
    print('j is %d' , j)
    print('k is %d' , k)

    return i, j, k


path = './testdata'
m = 0
n = 0
v = 0
model1 = CNN1(1, 10)
model2 = CNN2(1, 10)
model3 = CNN3(1, 10)
saved1 = torch.load('./mnistpkl/relu.pkl', map_location='cpu')
saved2 = torch.load('./mnistpkl/leakyrelu.pkl', map_location='cpu')
saved3 = torch.load('./mnistpkl/softplus.pkl', map_location='cpu')
model1.load_state_dict(saved1)
model2.load_state_dict(saved2)
model3.load_state_dict(saved3)
model1.eval()
model2.eval()
model3.eval()
criterion = nn.CrossEntropyLoss()
device = 'cpu'
for p in os.listdir(path):
    img = os.path.join(path, p)
    print(img)
    orig = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = orig.copy().astype(np.float32)

    IMG_SIZE = 28

    mean = [0.5]
    std = [0.5]
    img /= 255.0
    img = (img - mean)/std
    i, j , k = fun(img)
    m = m + i
    n = n + j
    v = v + k 


print(m)
print(n)
print(v)
