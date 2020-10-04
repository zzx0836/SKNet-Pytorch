import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from dataset.dataset import trainloader
from tensorboardX import SummaryWriter

from models.sknet import SKNet
from train import train_epoch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__=='__main__':



    net = SKNet(10)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-5, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().cuda()

    log_path = './logs/'
    writer = SummaryWriter(log_path)

    epoch_num = 300
    lr0 = 1e-3
    for epoch in range(epoch_num):
        current_lr = lr0 / 2 ** int(epoch / 50)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        train_epoch(net, optimizer, trainloader, criterion, epoch, writer=writer)

        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join('./model/model_{}.pth'.format(epoch)))
    torch.save(net.state_dict(), os.path.join('./model/model.pth'))

