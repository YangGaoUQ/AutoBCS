################  AutoBCS net implementation ########################
### import packages. 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
# import scipy.io as scio

class IntialReconNet(nn.Module):
    def __init__(self,SamplingPoints, BlockSize = 32):
        super(IntialReconNet, self).__init__()
        self.BlockSize = BlockSize  ## sliding block size, default: 32;
        self.SamplingPoints = SamplingPoints  ## sampling points (Ns) = round(sampling rate BlockSize**2)
        ## use dense convolutioanl layer to implement the sampling layer, y = FA * x
        self.sampling = nn.Conv2d(1, SamplingPoints , BlockSize, stride = 32, padding = 0,bias=False) 
        ## initialization, std: 0.028 is an empirical value from previous experiments 
        nn.init.normal_(self.sampling.weight, mean=0.0, std=0.028)  
        self.init_bl = nn.Conv2d(SamplingPoints, BlockSize ** 2, 1, bias=False) ## initial reconstruction
        # self.deep_bl = OctNet(2)  ## second part of the AutoBCS net

    def forward(self, x):
        device = torch.device("cpu")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nb = x.size(0)
        x = self.ImageCrop(x)     ## crop the image into patches of pre-defined blocksize
        x_bl = torch.zeros(nb, self.num_patches, self.SamplingPoints, 1, 1) 
        for i in range(0, self.num_patches):
            temp = x[:, i, :, :, :]
            temp = temp.to(device)
            temp = self.sampling(temp)   ## sampling block
            x_bl[:, i, :, :, :] = temp

        y_bl = torch.zeros(nb, self.num_patches, self.BlockSize ** 2, 1, 1)

        ## initial Recon
        for i in range(0, self.num_patches):
            temp_bl = x_bl[:,i,:,:,:]
            temp_bl = torch.squeeze(temp_bl, 1)
            temp_bl = temp_bl.to(device)  
            temp_bl = self.init_bl(temp_bl)    ## initial reconstructions block by block
            y_bl[:, i, :, :, :] = temp_bl

        ## reshape and concatenation. 
        y_bl = self.ReShape(y_bl)
        y_bl = y_bl.to(device)
        y_IR = y_bl
        ## OctNet reconstruction (if exists)
        # y_final = self.deep_bl(y_IR)
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
                #temp = torch.unsqueeze(temp, 1)
                #print(temp.size())
                temp2 = y[:,count,:,:,:,]
                #print(temp2.size())
                y[:,count,:,:,:,] = temp
                count = count + 1
        #print('Crop: %d'%count)
        #print(y.size())
        self.oriH = H
        self.oriL = L
        self.num_patches = num_patches
        return y

    def ReShape(self, x, BlockSize = 32):
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
        #print('reshape: %d'% count)
        return y


###################################
if __name__ == '__main__':
    yangcsnet = IntialReconNet(102)
    print(yangcsnet.state_dict)
    x = torch.randn(2, 1, 256, 256, dtype=torch.float)
    print('input ' + str(x.size()))
    print(x.dtype)
    y_IR = yangcsnet(x)
    print('output(1): '+str(y_IR.size()))