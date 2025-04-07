"""
--GM12878细胞系；11+11=22
DeepEPI: Predict Ehancer-Promoter interactions status using DNA sequence and histone so on of EPI pairs.
Copyright (C) 2022  Xuxiaoqiang.
"""
#test 卷积
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import  torch.nn.functional as F
#原计划-2dCNN;现计划1DCNN
# dconv = nn.ConvTranspose2d(in_channels=1, out_channels= 1,  kernel_size=2, stride=2, padding=1,output_padding=0, bias= False)
# init.constant(dconv.weight, 1)
# print(dconv.weight)
# input = Variable(torch.ones(1, 1, 2, 2))
# print(input.size())
# print(dconv(input).size())

#dodule design
#simple fully connected
class EPI_AutoEncoders(nn.Module):
    def __init__(self):
        super(EPI_AutoEncoders, self).__init__()
        # 定义Encoder编码层
        self.Encoder=nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1, stride=1),  # [B, 64, 3000]
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=4,stride=4),                                #[B,64,750]
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1,padding=1), # [B, 128, 752]
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=4, stride=4),  # [B,128,188]
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1),  # [B, 256, 188]
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=4, stride=4),  # [B,256,47]
        )
        self.fc_en=nn.Sequential(
            nn.Linear(256 * 47, 1024),
            nn.Linear(1024, 22)                      #22=11+11
        )
        self.fc_de = nn.Sequential(
            nn.Linear(22, 1024),
            nn.Linear(1024, 256 * 47)
        )
        self.Decoder = nn.Sequential(               # [B,256,47]
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=4, padding=0, output_padding=0),# [B,128,188]
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=4, padding=0, output_padding=0),  # [B,64,750]
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=4, padding=0, output_padding=0),  # [B,64,3000]
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 4, kernel_size=1, stride=1, padding=0, output_padding=0),  # [B,4,3000]
            nn.ReLU(True),
            nn.BatchNorm1d(4),
        )

    def forward(self,x):
        encoder=self.Encoder(x)
        encoder=encoder.view(-1,256*47)
        encoder_fc=self.fc_en(encoder)#22
        #print(encoder_fc.size())
        decoder_fc=self.fc_de(encoder_fc)#256*47
        decoder_fc=decoder_fc.unsqueeze(1)
        #print(decoder_fc.size())
        decoder=decoder_fc.view(-1,256,47)
        decoder=self.Decoder(decoder)
        #encoder_fc = torch.sigmoid(encoder_fc)
        return encoder_fc,decoder
#test tensor size of inout and output
# x = torch.randn(1, 4, 3000)
# EPIpred=EPI_AutoEncoders()
# FCout1,out2=EPIpred(x)
# print(FCout1.size())
# print(out2.size())