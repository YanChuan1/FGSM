import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN3, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(7*7*64, 200)
        self.fc2 = nn.Linear(200, self.num_classes)


    def forward(self, x):
    
        x = F.softplus(self.conv1_1(x), beta = 8)
        x = F.softplus(self.conv1_2(x), beta = 8)
    
        x = self.maxpool1(x)
    
        x = F.softplus(self.conv2_1(x), beta = 8)
        x = F.softplus(self.conv2_2(x), beta = 8)
    
        x = self.maxpool2(x)
    
        x = x.view(x.size(0), -1)
        x = F.softplus(self.fc1(x), beta = 8)
        x = self.fc2(x)
    
        return x


# if __name__ == '__main__':
#     saved = torch.load('9920.pth.tar', map_location='cpu')
#     model = CNN(1, 10)
#  #   model.load_state_dict(saved['state_dict'])
#    # saved = torch.load('relu.pkl')
#     model.load_state_dict(saved)

