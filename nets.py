import torch.nn as nn
import torch.nn.functional as F

mse = nn.MSELoss(reduction='none')

def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

class reshape(nn.Module):
    def __init__(self, *args):
        super(reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class _Cifar10_netI(nn.Module):
    def __init__(self, nz, nif=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, nif, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv51 = nn.Conv2d(nif * 8, nz, 4, 1, 0)  # for mu
        self.conv52 = nn.Conv2d(nif * 8, nz, 4, 1, 0)  # for log_sigma

    def forward(self, input):
        oI_l4 = self.main(input)
        oI_mu = self.conv51(oI_l4)
        oI_log_sigma = self.conv52(oI_l4)
        return oI_mu, oI_log_sigma

class _GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.bn_1 = nn.BatchNorm2d(self.in_channels)
        self.bn_2 = nn.BatchNorm2d(self.hidden_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        if self.learnable_sc:
            self.sc = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        h = x
        h = F.relu(self.bn_1(h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True) if self.upsample else h
        h = F.relu(self.bn_2(self.conv1(h)))
        h = self.conv2(h)

        y = x
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True) if self.upsample else y
        y = self.sc(y) if self.learnable_sc else y

        return h + y

class _Cifar10_netG(nn.Module):
    def __init__(self, nz, ngf=256, bottom_width=4):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.bottom_width = bottom_width

        self.lin_1 = nn.Linear(nz, (self.bottom_width**2)*self.ngf)
        self.block2 = _GenBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = _GenBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = _GenBlock(self.ngf, self.ngf, upsample=True)
        self.bn_5 = nn.BatchNorm2d(self.ngf)
        self.conv5 = nn.Conv2d(self.ngf, 3, 3, 1, 1)

    def forward(self, z):
        z = z.view(-1, z.shape[1])
        h = self.lin_1(z).view(-1, self.ngf, self.bottom_width, self.bottom_width) # 4x4
        h = self.block2(h) # 8x8
        h = self.block3(h) # 16x16
        h = self.block4(h) # 32x32
        h = F.relu(self.bn_5(h))
        h = self.conv5(h)
        h = F.tanh(h)

        return h

class EBMBlock(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channels=None, downsample=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channel
        self.learnable_sc = in_channel != out_channel or downsample
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channel, self.hidden_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(self.hidden_channels, out_channel, 3, 1, 1, bias=False)

        if self.learnable_sc:
            self.sc = nn.Conv2d(in_channel, out_channel, 1, bias=False)

    def forward(self, x):
        h = x
        h = F.relu(h)
        h = self.conv2(F.relu(self.conv1(h)))
        h = F.avg_pool2d(h, 2) if self.downsample else h

        y = x
        y = self.sc(y) if self.learnable_sc else y
        y = F.avg_pool2d(y, 2) if self.downsample else y
        return h + y

class EBMBlockStart(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channels=None, downsample=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channel
        self.learnable_sc = in_channel != out_channel or downsample
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channel, self.hidden_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(self.hidden_channels, out_channel, 3, 1, 1, bias=False)

        if self.learnable_sc:
            self.sc = nn.Conv2d(in_channel, out_channel, 1, bias=False)

    def forward(self, x):
        h = x
        h = F.relu(self.conv1(h))
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2) if self.downsample else h

        y = x
        y = F.avg_pool2d(y, 2) if self.downsample else y
        y = self.sc(y) if self.learnable_sc else y

        return h + y

class _Cifar10_netE(nn.Module):
    def __init__(self, nc, ndf, ):
        super().__init__()
        self.nef = ndf
        # Build the layers
        self.conv_1 = EBMBlockStart(nc, self.nef, downsample=True)
        self.block2 = EBMBlock(self.nef, self.nef, downsample=True)
        self.block3 = EBMBlock(self.nef, self.nef, downsample=False)
        self.block4 = EBMBlock(self.nef, self.nef, downsample=False)
        self.lin_5 = nn.Linear(self.nef, 1, bias=False)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = self.lin_5(x.view(x.shape[0], -1))
        return x