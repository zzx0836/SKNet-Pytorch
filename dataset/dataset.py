import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
])

trainset = torchvision.datasets.CIFAR10(root='DataSets/',train=True,download=True
                                        ,transform=transforms)
testset = torchvision.datasets.CIFAR10(root='DataSets/',train=False,
                                       download=True)
trainloader = DataLoader(trainset,batch_size=30,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=30,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')