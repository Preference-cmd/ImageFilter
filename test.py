import os
import torch
import numpy as np
import utils.denoise as de
from tqdm import trange


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
        v = de.DataFetch(load,save,batch_size,batch[k])
        tester = de.BatchTest(v)
        tester.forward()
        tester.save()
        torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()
    