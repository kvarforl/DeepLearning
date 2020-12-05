import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# get MNIST data
transforms = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])
                                # normalize to stdev and mean of MNIST dataset [-1,1]
train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data',
                                train=True, 
                                download=True,
                                transform=transforms), 
                batch_size=64, 
                shuffle=True)
test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms),
                batch_size=64, 
                shuffle=True)

#create first neural net (fully connected)
class FirstNet(nn.Module):
    #create all layers in init
    def __init__(self,image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 1000)
        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, 10)
    #layout relationship of layers in forward
    def forward(self, x):
        x = x.view(-1, self.image_size) #flattens tensor
        x = F.relu(self.fc0(x)) #activation function, max(x,0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x) #log_softmax puts range of values [0,1]

model = FirstNet(image_size=28*28)

optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10 + 1):
    train(epoch)
    test()