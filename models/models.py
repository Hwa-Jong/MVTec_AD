import torch.nn as nn

from .common import ConvResBlock, ConvBlock

class AE_student(nn.Module):
    def __init__(self):
        super(AE_student, self).__init__()
        self.channel = [3, 32]

        self.student = []

        for i in range(len(self.channel)-1):
            self.student.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i+1], kernel_size=3, bn=True, act=nn.ReLU()) )

        for i in range(len(self.channel)-1, 1, -1):
            self.student.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i-1], kernel_size=3, bn=True, act=nn.ReLU()) )

        self.student.append( nn.Conv2d(in_channels=self.channel[1], out_channels=self.channel[0], kernel_size=3, padding=1) )
        
        self.student = nn.Sequential(*self.student)
        
    def forward(self, x):  
        x = self.student(x)
        return x
    
