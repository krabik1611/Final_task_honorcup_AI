from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import cv2 as cv
import torch
import os


device =  "cpu"

class ResBlock(nn.Module):
    '''Class for implementing the residual block '''
    def __init__(self, inplanes:int=8, planes:int=8, stride:int=1)-> None:
        '''initializes base layers and functions'''
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''base pass residual block'''
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out



class Net(nn.Module):
    def __init__(self,block_size:int=3,stride:int=1)->None:
        '''initializes base layers and functions'''
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,8,kernel_size=block_size,padding=0,stride=stride)
        self.resBlock = ResBlock()
        self.conv2 = nn.Conv2d(8,8,2,stride=2,padding=1)
        self.conv3 = nn.Conv2d(8,3,kernel_size=block_size,padding=1,stride=stride)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x)
        x = self.resBlock(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x




class LearnDataset(Dataset):
    def __init__(self,root_dir:str,transforms:transforms=None)->None:
        '''Sets home directory and necessary transformations'''
        self.root_dir = root_dir
        self.transforms = transforms
        self.paths = self.__files__()

    def __files__(self) -> list:
        '''returns a list of files with the same name'''
        LR = "LR/"                                                                  # Low resolution folder in root_dir
        HR = "HR/"                                                                  # Hight  resolution folder in root_dir
        folder1 = os.listdir(self.root_dir+LR)                                      # Hight resolution files
        folder2 = os.listdir(self.root_dir+HR)                                      # Low  resolution files
        paths = []
        for file in folder1:                                                        # Find files with same names and append it in list
            if file in folder2:
                paths.append(file)
            else:
                continue
        return paths

    def __len__(self)->int:
        '''Returns the number of files'''
        return len(self.paths)

    def __getitem__(self,idx) -> list:
        '''returns a list of two photos: good and bad resolution'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1_name = os.path.join(self.root_dir,"HR/{}".format(self.paths[idx]))
        img2_name = os.path.join(self.root_dir,"LR/{}".format(self.paths[idx]))

        img1 = cv.imread(img1_name)
        img2 = cv.imread(img2_name)


        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        return [img1,img2]

class FinalDataset(Dataset):
    def __init__(self,root_dir:str,transforms:transforms=None)->None:
        '''Sets home directory and necessary transformations'''
        self.root_dir = root_dir
        self.transforms = transforms
        self.paths = self.__files__()

    def __files__(self) -> list:
        '''returns a list of files with the same name'''
        HR = "HR/"
        folder2 = os.listdir(self.root_dir+HR)

        return folder2

    def __len__(self):
        '''Returns the number of files'''
        return len(self.paths)

    def __getitem__(self,idx):
        '''returns a list with a photo and its name'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1_name = os.path.join(self.root_dir,"HR/{}".format(self.paths[idx]))

        img1 = cv.imread(img1_name)

        if self.transforms:
            img1 = self.transforms(img1)
        return [img1,self.paths[idx]]


def show(imgs:list) -> None:
    '''displays a list of photos via pyplot'''
    for i,img in enumerate(imgs,1):
        img : np.array = tensorToNumpy(img)
        ax = plt.subplot(1,len(imgs),i)
        ax.set_title(i)
        # ax.set_xticks([]),ax.set_yticks([])
        plt.imshow(img)
    plt.show()





def learn(epochs:int=3) -> None:
    '''function for training neural network'''
    learnroot = "../train/"                                                         # folder with LR and HR images
    learn_dataset = LearnDataset(learnroot,transforms.ToTensor())
    learn_dataloader = torch.utils.data.DataLoader(learn_dataset, batch_size=1,     # create dataloader
                                             shuffle=True, num_workers=1)
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)                         # create optimizer for neural network
    criterion = nn.L1Loss()                                                         # create L1Loss for neural network
    for epoch in range(epochs):
        for i,data in enumerate(learn_dataloader):
            optimizer.zero_grad()
            img1,img2 = data
            out = net(img1.to(device))
            loss = criterion(out,img2.to(device))
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Epoch:{epoch}\tLoss:{loss.item()}\tComplete:{i}/{len(learn_dataloader)}")
        torch.save(net.state_dict(),"model.th")                                     # saving every era so that there are no problems during accidental breakdowns


def tensorToNumpy(tensor:torch.Tensor)-> np.array:
    '''just converts the tensor to an array, so as not to write each time by hand'''
    return tensor.squeeze(0).cpu().detach().numpy().transpose(1,2,0)

def save(name:str,img:torch.Tensor)-> None:
    '''save Tensor to image'''
    img = tensorToNumpy(img)
    folder = "../test/LR/"
    fullname = folder+name
    cv.imwrite(fullname,img)


def run_final()-> None:
    '''function for the final creation of the necessary pictures'''
    final_dataset = FinalDataset("../test/",transforms.ToTensor())                  # create special dataset class :)
    dataloader = torch.utils.data.DataLoader(final_dataset, batch_size=1,
                                             shuffle=False, num_workers=1)
    net = Net().to(device)
    net.load_state_dict(torch.load('model.th'))                                     # load model
    with torch.no_grad():
        for data in dataloader:
            img1,name = data
            out = net(img1.to(device))*255
            save(name[0],out)




def run() -> None:
    '''function for visualizing the operation of a neural network'''
    dataroot = "../train/"                                                          # run on train dataset
    dataset = LearnDataset(dataroot,transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=1)
    net = Net().to(device)
    net.load_state_dict(torch.load('model.th'))

    with torch.no_grad():
        for data in dataloader:
            img1,img2 = data
            out = net(img1.to(device))
            show([img1,img2,out])                                                   # show 3 iamge: good, bad and downsample image


if __name__ == '__main__':
        learn(10)
        # run()
        run_final()
