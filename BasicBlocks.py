## import defaults packages 
import torch
import torch.nn as nn
import torch.nn.functional as F

## basic blocks
class OctEncodingBlocks(nn.Module):
    def __init__(self, num_in, num_out, alphax = 0.5, alphay = 0.5):
        super(OctEncodingBlocks, self).__init__()
        self.EncodeConv1 = OctConv(num_in, num_out, alphax, alphay)
        if alphax == 1:
            alphax = 0.5
        self.EncodeConv2 = OctConv(num_out, num_out, alphax, alphay)

    def forward(self, x_h, x_l):
        y_h, y_l = self.EncodeConv1(x_h, x_l)
        y_h, y_l = self.EncodeConv2(y_h, y_l)
        return y_h, y_l 

class OctMidBlocks(nn.Module):
    def __init__(self,  num_ch, alphax = 0.5, alphay = 0.5):
        super(OctMidBlocks, self).__init__()
        self.MidConv1 = OctConv(num_ch, 2 * num_ch, alphax, alphay)
        self.MidConv2 = OctConv(2 * num_ch, num_ch, alphax, alphay)

    def forward(self, x_h, x_l):
        y_h, y_l = self.MidConv1(x_h, x_l)
        y_h, y_l = self.MidConv2(y_h, y_l)
        return y_h, y_l 


class OctDecodingBlocks(nn.Module):
    def __init__(self,  num_in, num_out, alphax = 0.5, alphay = 0.5, bilinear=False):
        super(OctDecodingBlocks, self).__init__()
        if bilinear:
            self.up = OctUp(num_in, alphax = 0.5,  bilinear=True)
        else:
            self.up = OctUp(num_in, alphax = 0.5,  bilinear=False)
        self.DecodeConv1 = OctConv(2 * num_in, num_in, alphax, alphay)
        self.DecodeConv2 = OctConv(num_in, num_out, alphax, alphay)
          
    def forward(self, x_h1, x_l1, x_h2, x_l2):
        x_h1, x_l1 = self.up(x_h1, x_l1)
        x_h = torch.cat([x_h1, x_h2], dim = 1)
        x_l = torch.cat([x_l1, x_l2], dim = 1)
        y_h, y_l = self.DecodeConv1(x_h,x_l)
        y_h, y_l = self.DecodeConv2(y_h, y_l)
        return y_h, y_l


class OctUp(nn.Module):
    def __init__(self,num_in, alphax = 0.5,  bilinear=False):
        super(OctUp, self).__init__()
        self.In_H = int(num_in * alphax)
        self.In_L = (num_in - self.In_H)
        if bilinear:
            self.up_HH = nn.Upsample(scale_factor=2, mode='nearest')
            self.up_LH = nn.ConvTranspose2d(self.In_L, self.In_H, 4, stride = 4)
            self.BN_H = nn.BatchNorm2d(self.In_H)
            self.ReLU_H = nn.ReLU()
            self.up_LL = nn.Upsample(scale_factor=2, mode='nearest')
            self.up_HL = nn.Conv2d(self.In_H, self.In_L, 3, padding= 1)
            self.BN_L = nn.BatchNorm2d(self.In_L)
            self.ReLU_L = nn.ReLU()
        else:
            self.up_HH = nn.ConvTranspose2d(self.In_H, self.In_H, 2, stride = 2)
            self.up_LH = nn.ConvTranspose2d(self.In_L, self.In_H, 4, stride = 4)
            self.BN_H = nn.BatchNorm2d(self.In_H)
            self.ReLU_H = nn.ReLU()
            self.up_LL = nn.ConvTranspose2d(self.In_L, self.In_L, 2, stride = 2)
            self.up_HL = nn.Conv2d(self.In_H, self.In_L, 3, padding= 1)
            self.BN_L = nn.BatchNorm2d(self.In_L)
            self.ReLU_L = nn.ReLU()

    def forward(self, x_h, x_l):
        y_h1 = self.up_HH(x_h)
        y_h2 = self.up_LH(x_l)
        y_l1 = self.up_LL(x_l)
        y_l2 = self.up_HL(x_h)
        y_l = y_l1 + y_l2
        y_h = y_h1 + y_h2
        y_h = self.BN_H(y_h)
        y_h = self.ReLU_H(y_h)
        y_l = self.BN_L(y_l)
        y_l = self.ReLU_L(y_l)
        return y_h, y_l


class  OctConv(nn.Module):
    def __init__(self, num_in, num_out, alphax, alphay, ks= 3, pd = 1):
        super(OctConv, self).__init__()
        ## This calss Defines the OctConv operation. 
        ## define feature decomposition
        self.In_H = int(num_in * alphax)
        self.In_L = (num_in - self.In_H)
        self.Out_H = int(num_out * alphay)
        self.Out_L = (num_out - self.Out_H)
        self.alphax = alphax
        self.alphay = alphay
        ## define operations 
        if alphax == 1:
            ## for the input layer.
            self.convHH = nn.Conv2d(self.In_H, self.Out_H , ks, padding=pd)
            self.convHL = nn.Conv2d(self.In_H, self.Out_L , ks, padding=pd)
            self.BN_HH = nn.BatchNorm2d(self.Out_H)
            self.BN_HL = nn.BatchNorm2d(self.Out_L)
            self.ReLU_H = nn.ReLU()
            self.ReLU_L = nn.ReLU()
        elif alphay == 1:
            ## only for the final output layer
            self.convHH = nn.Conv2d(self.In_H, self.Out_H , ks, padding=pd)
            self.convLH = nn.Conv2d(self.In_L, self.Out_H , ks, padding=pd)
            self.up = nn.ConvTranspose2d(self.Out_H, self.Out_H, 2, stride = 2)
            self.BN_HH = nn.BatchNorm2d(self.Out_H)
            self.BN_LH = nn.BatchNorm2d(self.Out_H)
            self.ReLU_H = nn.ReLU()
            self.FinalConv = nn.Conv2d(self.Out_H, 1, 1, stride = 1, padding = 0)
        else:
            ## mid layers
            self.convHH = nn.Conv2d(self.In_H, self.Out_H , ks, padding=pd)
            self.convLL = nn.Conv2d(self.In_L, self.Out_L , ks, padding=pd)
            self.convHL = nn.Conv2d(self.In_H, self.Out_L , ks, padding=pd)
            self.convLH = nn.Conv2d(self.In_L, self.Out_H , ks, padding=pd)
            self.up = nn.ConvTranspose2d(self.Out_H, self.Out_H, 2, stride = 2)
            self.BN_HH = nn.BatchNorm2d(self.Out_H)
            self.BN_LH = nn.BatchNorm2d(self.Out_H)
            self.ReLU_H = nn.ReLU()
            self.BN_LL = nn.BatchNorm2d(self.Out_L)
            self.BN_HL = nn.BatchNorm2d(self.Out_L)
            self.ReLU_L = nn.ReLU()

    def forward(self, x_h, x_l):
        ## Y_H = conv(H) + upsample(conv(L))
        ## Y_L = conv(L) + conv(avgpool(H))
        if self.alphax == 1:
            y_h = self.convHH(x_h)
            y_h = self.BN_HH(y_h)
            y_l = F.avg_pool2d(x_h, 2)
            y_l = self.convHL(y_l)
            y_l = self.BN_HL(y_l)
            ## BN and ReLU()
            y_h = self.ReLU_H(y_h)
            y_l = self.ReLU_L(y_l)
            return y_h, y_l
        elif self.alphay == 1:
            y_h1 = self.convHH(x_h)
            y_h1 = self.BN_HH(y_h1)
            y_h2 = self.convLH(x_l)
            y_h2 = self.up(y_h2)
            y_h2 = self.BN_LH(y_h2)
            ## BN and ReLU()
            y_h = y_h1 + y_h2
            y_h = self.ReLU_H(y_h)
            ## final Output, Convolution without relu, 
            y_h = self.FinalConv(y_h)
            return y_h
        else:
            ## high fre group
            ## first convLH then Upsample
            y_h1 = self.convHH(x_h)
            y_h2 = self.convLH(x_l)
            y_h2 = self.up(y_h2)
            y_h1 = self.BN_HH(y_h1)
            y_h2 = self.BN_LH(y_h2)
            ## low fre group
            ## first pooling, then convolution
            y_l1 = self.convLL(x_l)
            y_l2 = F.avg_pool2d(x_h, 2)
            y_l2 = self.convHL(y_l2)
            y_l1 = self.BN_LL(y_l1)
            y_l2 = self.BN_HL(y_l2)
            ## final addition
            y_h = y_h1 + y_h2
            y_l = y_l1 + y_l2
            ## BN and ReLU()
            y_h = self.ReLU_H(y_h)
            y_l = self.ReLU_L(y_l)
            return y_h, y_l