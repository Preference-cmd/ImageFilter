import os
import torch
import cv2
import os
import torchvision
import numpy as np
import torchvision.transforms as transform
from models.network_dncnn import DnCNN as net
from tqdm import trange


class BatchTest:
    
    # ----------------------------------------------------------------
    # Copyright
    # @Preference
    # 
    # device: current device that model is working on
    # dataloader: a class that implements loading and saving images
    # model: denoisor
    #
    # ----------------------------------------------------------------
    
    def __init__(self, data):
        self.device = torch.device(                          
            'cuda' if torch.cuda.is_available() else 'cpu')  
        self.dataloader = data                                                           
        self.denoisedImg = []                                
        self.model = self.loadNetwork()                       
                                                             
    def forward(self):
        imgs = [i.to(self.device) for i in self.dataloader.img]
        with torch.no_grad():
            for pic in imgs:
                pic = torch.split(pic,1,dim=0)
                pic = torch.stack(
                    [self.model(pic[0]),self.model(pic[1]),self.model(pic[2])],
                    dim=1
                )
                self.denoisedImg.append(pic)

    def loadNetwork(self):
        model = net(1, 1, 64, 20, 'R')
        model.load_state_dict(torch.load('dncnn3.pth'))
        model.to(self.device)
        model.eval()
        return model
    
    def save(self):
        self.dataloader.saveImg(self.denoisedImg)



class DataFetch:
    def __init__(self, loadDir, saveDir, miniBatchSize, startIndex):
        self.workDir = [loadDir, saveDir]
        self.imgName = os.listdir(loadDir)[startIndex:]
        self.dir = self.select(miniBatchSize)
        self.img = self.loadImg()
        #print(self.imgName)
        #print(self.dir)
        
    def __iter__(self):
        return self.img
    
    def loadImg(self):
        imgs = []
        for each in self.dir:
            img = cv2.imread(f'{self.workDir[0]}/{each}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform.ToTensor()(img)
            imgs.append(img)
        return imgs
    
    def saveImg(self,imgs):
        for k in range(len(imgs)):
            torchvision.utils.save_image(imgs[k], f'{self.workDir[1]}/{self.dir[k]}')
        
    def select(self, batch):
        if len(self.imgName) < batch:
            dirs = self.imgName
        else:
            dirs = self.imgName[:batch]
        #print(dirs)
        return dirs


def calStart(batchSize, loadPath):
    num = len(os.listdir(loadPath))
    temp = int(num / batchSize)
    startList = []
    start = 0
    if np.mod(num,batchSize) > 0:
        batchNum = temp + 1
    else:
        batchNum = temp
    for i in range(batchNum):
        startList.append(start)
        start = start + batchSize
    return startList
    
def main():
    load = 'Noised'
    save = 'Denoised/dncnn'
    batch_size = 1
    batch = calStart(batch_size, load)
    
    for k in trange(len(batch)):
        v = DataFetch(load,save,batch_size,batch[k])
        tester = BatchTest(v)
        tester.forward()
        tester.save()
        torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()
    