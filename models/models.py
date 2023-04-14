import torch.nn as nn

from .common import Upsample, ConvBlock

class AE_student(nn.Module):
    def __init__(self):
        super(AE_student, self).__init__()
        self.channel = [3, 8, 16, 32]

        self.encoder = []
        self.decoder = []

        # encoder
        self.encoder.append( ConvBlock(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=3, bn=True, act=nn.ReLU()) )
        self.encoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, bn=True, act=nn.ReLU()) )
        self.encoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, bn=True, stride=2, act=nn.ReLU()) )

        for i in range(1, len(self.channel)-1):
            self.encoder.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i+1], kernel_size=3, bn=True, act=nn.ReLU()) )
            self.encoder.append( ConvBlock(in_channels=self.channel[i+1], out_channels=self.channel[i+1], kernel_size=3, bn=True, stride=2, act=nn.ReLU()) )


        # decoder
        for i in range(len(self.channel)-1, 1, -1):
            self.decoder.append( ConvBlock(in_channels=self.channel[i], out_channels=self.channel[i], kernel_size=3, bn=True, act=nn.ReLU()) )
            self.decoder.append( Upsample(channels=self.channel[i], out_channels=self.channel[i-1], scale=2) )

        self.decoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, bn=True, act=nn.ReLU()) )
        self.decoder.append( Upsample(channels=self.channel[1], out_channels=self.channel[1], scale=2) )

        self.decoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, bn=True, act=nn.ReLU()) )
        self.decoder.append( ConvBlock(in_channels=self.channel[1], out_channels=self.channel[1], kernel_size=3, bn=True, act=nn.ReLU()) )
        self.decoder.append( nn.Conv2d(in_channels=self.channel[1], out_channels=self.channel[0], kernel_size=3, padding=1) )
        
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):  
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
