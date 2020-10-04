import torch
from torch import nn


class ResNeXUnit(nn.Module):
    def __init__(self,in_features,out_featuares,mid_features=None,stride=1,groups=32):
        super(ResNeXUnit,self).__init__()
        if mid_features is None:
            mid_features = int(out_featuares/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features,mid_features,1,stride=1),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features,mid_features,3,stride=stride,padding=1,groups=groups),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features,out_featuares,1,stride=1),
            nn.BatchNorm2d(out_featuares)
        )
        if in_features==out_featuares:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features,out_featuares,1,stride=stride),
                nn.BatchNorm2d(out_featuares)
            )
    def forward(self,x):
        fea = self.feas(x)
        return fea+self.shortcut(x)
class ResNetxt(nn.Module):
    def __init__(self,class_num):
        super(ResNetxt, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64)
        )
        self.stage_1 = nn.Sequential(
            ResNeXUnit(64,256,mid_features=128),
            nn.ReLU(),
            ResNeXUnit(256,256),
            nn.ReLU(),
            ResNeXUnit(256,256),
            nn.ReLU()
        )
        self.stage_2 = nn.Sequential(
            ResNeXUnit(256,512,stride=2),
            nn.ReLU(),
            ResNeXUnit(512,512),
            nn.ReLU(),
            ResNeXUnit(512,512),
            nn.ReLU()
        )
        self.stage_3 = nn.Sequential(
            ResNeXUnit(512,1024,stride=2),
            nn.ReLU(),
            ResNeXUnit(1024,1024),
            nn.ReLU(),
            ResNeXUnit(1024,1024),
            nn.ReLU()

        )
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024,class_num)
        )
    def forward(self,x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea
# if __name__=='__main__':
#     x = torch.rand(8,3,32,32)
#     net = ResNetxt(10)
#     out = net(x)
#     print(out.shape)
#     print(out)

