
############### predict ###################
import torch 
import torch.nn as nn
import numpy as np
import time
import nibabel as nib
import scipy.io as scio

import sys

sys.path.append('../Model')

from LSM_and_Initial_Recon import *
from OctNet import * 
##########################################

if __name__ == '__main__':
    with torch.no_grad():   
        print('OCT')
        ## load trained network 

        IniReconNet = LSM_IniReconNet(256) ## scaling factor 4; 
        IniReconNet = nn.DataParallel(IniReconNet)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        IniReconNet.load_state_dict(torch.load('../Pre-TrainedModel/IniReconNet_ScalingFactor_4.pth'))  
        IniReconNet.to(device)
        IniReconNet.eval()

        DeepOctNet = OctNet(2)
        DeepOctNet = nn.DataParallel(DeepOctNet)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        DeepOctNet.load_state_dict(torch.load('../Pre-TrainedModel/DeepOctNet_ScalingFactor_4.pth'))
        DeepOctNet.to(device)
        DeepOctNet.eval()

        File_No = 5 
        Folder_name = 'set5'
        for i in range(1, File_No + 1):     
            fname = ('../%s/(%d).mat'% (Folder_name, i))
            data = scio.loadmat(fname)
            image = data['img_label']
            print(fname)
            image = np.array(image)
            ## convert the image data to torch.tesors and return. 
            image = torch.from_numpy(image) 

            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image, 0)
            image = image.float()
            ################ Evaluation ##################
            image = image.to(device)
            pred_IR = IniReconNet(image)
            start_time = time.time()
            pred_FR = DeepOctNet(pred_IR)
            end_time = time.time()
            print(end_time - start_time)

            pred_IR = torch.squeeze(pred_IR, 0)
            pred_IR = torch.squeeze(pred_IR, 0)

            print(get_parameter_number(IniReconNet))
            pred_IR = pred_IR.to('cpu')
            pred_IR = pred_IR.numpy()

            path =  ('../%s/(%d)_initRecon.mat'% (Folder_name, i))
            scio.savemat(path, {'PRED_IR':pred_IR})

            print(pred_FR.size())
            pred_FR = torch.squeeze(pred_FR, 0)
            pred_FR = torch.squeeze(pred_FR, 0)

            pred_FR = pred_FR.to('cpu')
            pred_FR = pred_FR.numpy()

            path =  ('../%s/(%d)_finalRecon.mat'% (Folder_name, i))
            scio.savemat(path, {'PRED_FR':pred_FR})