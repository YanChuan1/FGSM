""" Fast Gradient Sign Method
    Paper link: https://arxiv.org/abs/1607.02533

    Controls:
        'esc' - exit
         's'  - save adversarial image
"""
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import numpy as np
import cv2
import argparse
from net import CNN


parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/img_3.jpg', help='path to image')
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--gpu', action="store_true", default=False)

args = parser.parse_args()
image_path = args.img
y_true = args.y
gpu = args.gpu

IMG_SIZE = 28

print('Fast Gradient Sign Method')
print()


def nothing(x):
    pass

window_adv = 'adversarial image'
cv2.namedWindow(window_adv, cv2.WINDOW_FREERATIO)
cv2.createTrackbar('eps', window_adv, 1, 255, nothing)


orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
img = orig.copy().astype(np.float32)
perturbation = np.empty_like(orig)

mean = [0.5]
std = [0.5]
img /= 255.0
img = (img - mean)/std


# load model
model1 = CNN(1, 10)

saved1 = torch.load('relu.pkl', map_location='cpu')

model1.load_state_dict(saved1)

model1.eval()


criterion = nn.CrossEntropyLoss()
device = 'cuda' if gpu else 'cpu'

# prediction before attack
inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)

out1 = model1(inp)


pred1 = np.argmax(out1.data.cpu().numpy())

print('Prediction before attack: %s' %(pred1))



while True:
    # get trackbar position
    eps = cv2.getTrackbarPos('eps', window_adv)
    inp1 = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
 

    out1 = model1(inp1)


    loss1 = criterion(out1, Variable(torch.Tensor([float(pred1)]).to(device).long()))

    # compute gradients
    loss1.backward()

    # this is it, this is the method
    inp1.data = inp1.data + ((eps/255.0) * torch.sign(inp1.grad.data))
    inp1.data = inp1.data.clamp(min=-1, max=1)
    inp1.grad.data.zero_() # unnecessary

    

    # predict on the adversarial image
    pred_adv1 = np.argmax(model1(inp1).data.cpu().numpy())

    print(" "*60, end='\r') # to clear previous line, not an elegant way
    print("After attack: eps [%f] \t%s" % (eps, pred_adv1), end="\r")  # , end='\r')#'eps:', eps, end='\r')

    # reprocess image
    adv1 = inp1.data.cpu().numpy()[0][0]
    perturbation1 = adv1-img
    adv1 = (adv1 * std) + mean
    adv1 = adv1 * 255.0
    adv1 = np.clip(adv1, 0, 255).astype(np.uint8)
    perturbation1 = perturbation1*255
    perturbation1 = np.clip(perturbation1, 0, 255).astype(np.uint8)


    # display images
    cv2.imshow('perturbation', perturbation1)
    cv2.imshow('adv', adv1)
    key = cv2.waitKey(500) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('./images/results/img_adv.png', adv1)
        cv2.imwrite('./images/results/perturbation.png', perturbation1)

print()
cv2.destroyAllWindows()
