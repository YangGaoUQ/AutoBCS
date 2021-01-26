################### train AutoBCS framework #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from DataLoad import *

sys.path.append('../Model')

from LSM_and_Initial_Recon import *
from OctNet import * 

#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):
    DATA_DIRECTORY = '../trainPatch'
    DATA_LIST_PATH = './test_IDs.txt'
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def SaveNet(IniReconNet, octReconNet):
    print('save results')
    torch.save(IniReconNet.state_dict(), './IniReconNet_100EPO_64BATCH_ScalingFactor_4.pth')
    torch.save(octReconNet.state_dict(), './DeepOctNet_100EPO_64BATCH_ScalingFactor_4.pth')

def TrainNet(IniReconNet, octReconNet, LR = 0.001, Batchsize = 32, Epoches = 40 , useGPU = True):
    print('DataLoad')
    trainloader = DataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.L1Loss()
    optimizer1 = optim.Adam(IniReconNet.parameters())
    optimizer2 = optim.Adam(octReconNet.parameters())
    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [50, 80], gamma = 0.1)
    scheduler2 = LS.MultiStepLR(optimizer2, milestones = [50, 80], gamma = 0.1)

    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            IniReconNet = nn.DataParallel(IniReconNet)
            IniReconNet.to(device)
            octReconNet = nn.DataParallel(octReconNet)
            octReconNet.to(device)
            for epoch in range(1, Epoches + 1):
                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    Inputs, Name = data
                    Inputs = Inputs.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    ## forward: 
                    Pred_IR = IniReconNet(Inputs)
                    pred_bl = octReconNet(Pred_IR)
                    ## loss
                    loss1 = criterion(Pred_IR, Inputs)
                    loss2 = criterion(pred_bl, Inputs)
                    ## backward
                    loss1.backward(retain_graph = True)
                    loss2.backward()
                    ##
                    optimizer1.step()
                    optimizer2.step()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 19 == 0:
                        acc_loss1 = loss1.item()   
                        acc_loss2 = loss2.item()   
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, loss2 : %f, lr1: %f, lr2: %f, used time: %d s' %
                            (epoch, i + 1, acc_loss1, acc_loss2, optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'], time_end - time_start))  
                scheduler1.step()
                scheduler2.step()
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    SaveNet(IniReconNet, octReconNet)

if __name__ == '__main__':
    ## data load
    ## create network 
    IniReconNet = LSM_IniReconNet(256)
    octReconNet = OctNet(2)
    print(IniReconNet.state_dict)
    print(octReconNet.state_dict)
    print(get_parameter_number(IniReconNet))
    print(get_parameter_number(octReconNet))
    ###### use this line to check if all layers 
    ###### are leanrable in this programe. 
    print('100EPO IniReconNet')
    ## train network
    TrainNet(IniReconNet,octReconNet, LR = 0.001, Batchsize = 64, Epoches = 100 , useGPU = True)

