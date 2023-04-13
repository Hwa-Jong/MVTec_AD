import torch.nn as nn

class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True, bn=True, act=nn.ReLU()):
        super(ConvBlock, self).__init__()

        self.conv_block = []

        self.conv_block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, bias=bias))
        if bn:
            self.conv_block.append(nn.BatchNorm2d(num_features=out_channels))
        if act is not None:
            self.conv_block.append(act)

        self.conv_block = nn.Sequential(*self.conv_block)
        

    def forward(self, x):  
        x = self.conv_block(x)
        return x
    


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=True, bn=True, act=nn.ReLU()):
        super(ConvResBlock, self).__init__()

        self.conv_block = []

        self.conv_block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=dilation, bias=bias))
        if bn:
            self.conv_block.append(nn.BatchNorm2d(num_features=out_channels))
        if act is not None:
            self.conv_block.append(act)

        self.conv_block.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=dilation, bias=bias))

        self.conv_block = nn.Sequential(*self.conv_block)
        

    def forward(self, x):  
        res = x
        x = self.conv_block(x)
        return x + res
    