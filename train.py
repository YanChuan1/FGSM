import torch
import net3
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义一些超参数
batch_size = 64
learning_rate = 0.02
num_epoches = 50

# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

# 数据集的下载器
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 选择模型
model = net3.CNN3(1, 10)
# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
epoch = 0
for data in train_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        img = Variable(img)
        label = Variable(label)

    # out1 = model(img)
    # loss1 = criterion(out1, label)
    # l1 = loss1
    # loss1.backward(retain_graph = True)

    # img.data = img.data + 0.007 * torch.sign(img.grad.data)
    # img.data = img.data.clamp(min=-1, max=1)
    # img.grad.data.zero_() 

    # out2 = model(img)
    # l2 = criterion(out2 ,label)
    # loss = 0.5 * l1 + 0.5 * l2
    #上面注释的部分是对抗训练
    out = model(img)
    loss = criterion(out, label)
    print_loss = loss.data.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch+=1
    if epoch%50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

torch.save(model.state_dict(), 'softplus.pkl')