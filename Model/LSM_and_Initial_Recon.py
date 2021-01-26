################  LSM and Intial reconstruction moduel ########################
### import packages. 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import scipy.io as scio
class LSM_IniReconNet(nn.Module):
    def __init__(self, SamplingPoints, BlockSize = 32):
        ## Smpling Points  = sampling ratios * 32 ** 2 
        super(LSM_IniReconNet, self).__init__()
        self.BlockSize = BlockSize
        self.SamplingPoints = SamplingPoints

        ## learnable LSM
        self.sampling = nn.Conv2d(1, SamplingPoints , BlockSize, stride = 32, padding = 0,bias=False)
        nn.init.normal_(self.sampling.weight, mean=0.0, std=0.028)  

        ## linear intial recon (by basic linear operator)
        self.init_bl = nn.Conv2d(SamplingPoints, BlockSize ** 2, 1, bias=False)

    def forward(self, x):
        ## cut the image into patches of pre-defined blocksize
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nb = x.size(0)

        x = self.ImageCrop(x)

        x_IR = torch.zeros(nb, self.num_patches, self.SamplingPoints, 1, 1)

        for i in range(0, self.num_patches):
            temp = x[:, i, :, :, :]
            temp = temp.to(device)
            temp = self.sampling(temp)
            x_IR[:, i, :, :, :] = temp

        y_IR = torch.zeros(nb, self.num_patches, self.BlockSize ** 2, 1, 1)

        ## initial Recon
        for i in range(0, self.num_patches):
            temp_IR = x_IR[:,i,:,:,:]
            temp_IR = torch.squeeze(temp_IR, 1)
            temp_IR = temp_IR.to(device)
            temp_IR = self.init_bl(temp_IR)
            y_IR[:, i, :, :, :] = temp_IR

        ## reshape and concatenation. 
        y_IR = self.Reshape(y_IR)
        y_IR = y_IR.to(device)

        return y_IR

    def ImageCrop(self, x, BlockSize = 32):
        H = x.size(2)
        L = x.size(3)
        nb = x.size(0)
        nc = x.size(1)
        num_patches = H * L // (BlockSize ** 2)
        y = torch.zeros(nb, num_patches, nc, BlockSize, BlockSize)
        ind1 = range(0, H, BlockSize)
        ind2 = range(0, L, BlockSize)
        count = 0
        for i in ind1:
            for j in ind2:
                temp = x[:,:,i:i+ BlockSize, j:j+BlockSize]
                temp2 = y[:,count,:,:,:,]
                y[:,count,:,:,:,] = temp
                count = count + 1
        self.oriH = H
        self.oriL = L
        self.num_patches = num_patches
        return y

    def Reshape(self, x, BlockSize = 32):
        nb = x.size(0)
        y = torch.zeros(nb, 1, self.oriH, self.oriL)
        ind1 = range(0, self.oriH, BlockSize)
        ind2 = range(0, self.oriL, BlockSize)
        count = 0
        for i in ind1:
            for j in ind2:
                temp = x[:,count,:,:,:]
                temp = torch.squeeze(temp, 1)
                temp = torch.reshape(temp, [nb, 1, BlockSize, BlockSize])
                y[:,:,i:i+BlockSize, j:j+BlockSize] = temp
                count = count + 1
        return y


###################################
if __name__ == '__main__':
    autobcs =  LSM_IniReconNet(256)
    x = torch.randn(2, 1, 256, 256, dtype=torch.float)
    print('input ' + str(x.size()))
    print(x.dtype)
    y = autobcs(x)
    print('output: '+str(y.size()))