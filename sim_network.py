import  torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

#创建模型
class NN(nn.Module):
    def __init__(self,input_size,num_classes): # 28*28
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#设置训练设备
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
#设置超参数
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

#加载数据
train_dataset= datasets.MNIST(root='./dataset',train=True, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_dataset= datasets.MNIST(root='./dataset',train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)


#初始化网络
model = NN(input_size=input_size,num_classes=num_classes).to(device)

#损失函数和优化
creiterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#训练网络
for epoch in range(num_epochs):
    for batch_idx ,(data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0],-1)

        scores = model(data)
        loss = creiterion(scores,targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



#测试模型的准确度
def check_accuracy(loader,model):
    if loader.dataset.train:
        print("training data")
    else:
        print("test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            score = model(x)
            _ , predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


    model.train()


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)