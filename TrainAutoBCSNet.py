################### train AutoBCSnet ###################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from InitiaReconNet import *
from OctNet import * 
from DataLoad import *
#########  Section 1: DataSet Load #############
def yangDataLoad(Batch_size):
    DATA_DIRECTORY = './trainPatch'  
    DATA_LIST_PATH = './test_IDs.txt' 
    dst = yangDataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def yangSaveNet(initNet, deepReconNet, enSave = False):
    print('save results')
    #### save the
    if enSave:
        torch.save(initNet, './yangEntireinitNet.pth')
    else:
        torch.save(initNet.state_dict(), './yanginitNet_100EPO_64BATCH_ACC_3_l1.pth')
        torch.save(deepReconNet.state_dict(), './yangDeepReconNET_100EPO_64BATCH_ACC_3_l1.pth')

def yangTrainNet(initNet, deepReconNet, LR = 0.001, Batchsize = 32, Epoches = 40 , useGPU = True):
    print('DataLoad')
    trainloader = yangDataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.L1Loss()  ## L1 loss function. use MSEloss() for L2 loss
    optimizer1 = optim.Adam(initNet.parameters())
    optimizer2 = optim.Adam(deepReconNet.parameters()) 
    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [50, 80], gamma = 0.1) # schedular for first subnet
    scheduler2 = LS.MultiStepLR(optimizer2, milestones = [50, 80], gamma = 0.1) # schedualr for second part

    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            initNet = nn.DataParallel(initNet)
            initNet.to(device)
            deepReconNet = nn.DataParallel(deepReconNet)
            deepReconNet.to(device)
            for epoch in range(1, Epoches + 1):
                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    Inputs, Name = data
                    Inputs = Inputs.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    ## network forward: 
                    Pred_IR = initNet(Inputs)      ## intial recon
                    pred_bl = deepReconNet(Pred_IR) ## octave recon
                    ## loss calculation
                    loss1 = criterion(Pred_IR, Inputs)
                    loss2 = criterion(pred_bl, Inputs)
                    ## backward
                    ## optimize whole network using loss1 and loss2
                    loss1.backward(retain_graph = True)  
                    loss2.backward()
                    ## updata parameters
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
                scheduler1.step()  ## update schedular1
                scheduler2.step()  ## update schedular2
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    yangSaveNet(initNet, deepReconNet)

if __name__ == '__main__':
    ## create network 
    """
    the AutoBCS net consists of two individual parts, i.e., initial reconstruction sub-network 
    and octave convolutional netowkrs. In this programme, we use seperate training optimizer to 
    optimize corresponding sub-networks, such design can make the training more flexibel (e.g, 
    optimizer, loss function, learning scheduler)
    """
    initNet = IntialReconNet(307)  ## first part of AutoBCS net: intial recon 
    deepReconNet = OctNet(2)       ## second part of AutoBcS net: octave network
    print(initNet.state_dict)
    print(deepReconNet.state_dict)
    print(get_parameter_number(initNet))
    print(get_parameter_number(deepReconNet))
    ###### use this line to check if all layers 
    ###### are leanrable in this programe. 
    print('100EPO Training')
    ## train network
    yangTrainNet(initNet,deepReconNet, LR = 0.001, Batchsize = 64, Epoches = 100 , useGPU = True)

