import torch
import torch.nn as nn

from .common import Upsample, ConvBlock

class AE_student(nn.Module):
    def __init__(self):
        super(AE_student, self).__init__()
        self.channel = [3, 8, 16, 32]

        self.encoder = []
        self.decoder = []

        # encoder
        for i in range(len(self.channel)-1):
            self.encoder.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i], kernel_size=3, bn=True, act=nn.ReLU()) )
            self.encoder.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i+1], kernel_size=3, bn=True, stride=2, act=nn.ReLU()) )


        # decoder
        for i in range(len(self.channel)-1, 1, -1):
            self.decoder.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i], kernel_size=3, bn=True, act=nn.ReLU()) )
            self.decoder.append( Upsample(channels=self.channel[i], out_channels=self.channel[i-1], scale=2) )

        self.decoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, bn=True, act=nn.ReLU()) )
        self.decoder.append( Upsample(channels=self.channel[1], out_channels=self.channel[1], scale=2) )

        self.decoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, bn=True, act=nn.ReLU()) )
        self.decoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[0], kernel_size=3, bn=True, act=nn.Sigmoid()) )
        
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):  
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class AE_v1(nn.Module):
    def __init__(self, input_shape):
        super(AE_v1, self).__init__()
        self.channel = [3, 8, 16]

        self.latent_feature = nn.parameter.Parameter(torch.rand(size=(1, self.channel[-1], input_shape[0], input_shape[1])).to(torch.float32))

        self.encoder = []
        self.decoder = []

        # encoder
        for i in range(len(self.channel)-1):
            self.encoder.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i+1], kernel_size=3, bn=True, act=nn.ReLU(True)) )

        # decoder
        for i in range(len(self.channel)-1, 1, -1):
            self.decoder.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i-1], kernel_size=3, bn=True, act=nn.ReLU(True)) )

        self.decoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[0], kernel_size=3, bn=True, act=nn.Sigmoid()) )
        
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):  
        x = self.encoder(x)
        diff = x - self.latent_feature
        x = self.decoder(x)
        return x, diff
    
