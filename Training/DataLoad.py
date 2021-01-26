import os
import numpy as np
import nibabel as nib
import random
import torch
import scipy.io as scio
from torch.utils import data

 
class DataSet(data.Dataset):
    def __init__(self, root, list_path):
        super(DataSet,self).__init__()

        ## root: data path;
        ## list_path: path for image_ids file

        self.root = root
        self.list_path = list_path
 
        ## get the number of files. 
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # print(self.img_ids)
        ## get all fil names, preparation for get_item. 
        ## for example, we have two files: 
        ## 102-input.mat for input, and 102-label.matfor label; 
        ## then image id is 102, and then we can use string operation
        ## to get the full name of the input and label files. 
        self.files = []
        for name in self.img_ids:
            img_file = self.root + ("/(%s).mat" % name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)
 
 
    def __getitem__(self, index):
        datafiles = self.files[index]
 
        '''load the datas'''
        name = datafiles["name"]
        ## read data. 
        data = scio.loadmat(datafiles["img"])
        image = data['subim_input']

        iamge = np.array(image)

        ## convert the image data to torch.tesors and return. 
        image = torch.from_numpy(image) 

        image = torch.unsqueeze(image, 0)

        image = image.float()
        
        return image, name
 
## before formal usage, test the validation of data loader. 
if __name__ == '__main__':
    DATA_DIRECTORY = './trainData'
    DATA_LIST_PATH = './test_IDs.txt'
    Batch_size = 4
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print(dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=False, drop_last = True)
    for i, Data in enumerate(trainloader):
        imgs, labels, aff, names = Data
        print(i)
        if i%1 == 0:
            print(names)
            print(aff.size)
            print(imgs.size())
            print(labels.size())