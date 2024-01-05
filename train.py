# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:52:10 2021

@author: zhangbowen
"""

import argparse 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import DatasetFromFolder
from model import SRCNN,AttentionZBW

#parser = argparse.ArgumentParser(description='SRCNN training parameters')
#parser.add_argument('--zoom_factor', type=int, required=True)
#parser.add_argument('--nb_epochs', type=int, default=200)
#parser.add_argument('--cuda', action='store_true')
#args = parser.parse_args()

# Parameters
# model_name = 'srcnn'
model_name='attentionzbw'
num_workers = 0  # windows: 0 is　recommended, linux: 4*num_GPU is recommended (for training speed)
scale = 3
batch_size = 4
epochs = 1000
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# dataset
train_set = DatasetFromFolder("data/train", scale)
test_set = DatasetFromFolder("data/test", scale)
print('train set:',train_set.__len__())
print('test set:',test_set.__len__())
print('----------')

# dataloader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# check device
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device：", device)
print('----------')


##### load SRCNN & AttentionZBW #####
if model_name == 'srcnn':
    model = SRCNN()
    print("Model：",model)
    print('----------')
    model.to(device)
elif model_name == 'attentionzbw':
    model = AttentionZBW()
    print("Model：", model)
    print('----------')
    model.to(device)


# loss function
criterion = nn.MSELoss()

# optimization
# Adam is faster than SGD
optimizer = optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4)

# train & test(PSNR)
for epoch in range(epochs):

    ###### train #####
    epoch_loss = 0
    for iteration, batch in enumerate(train_loader):

        # forward
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad() 
        outputs = model(inputs)
   
        # backward
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('----------') 
    print('Epoch %d--Training Loss: %.4f' % (epoch+1,epoch_loss/(iteration+1)))

    ###### test #####
    avg_psnr = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # evaluate by PSNR
            psnr = 10 * torch.log10(1*1 / loss)
            avg_psnr += psnr
    print('Average PSNR: %.4f[dB]' % (avg_psnr / len(test_loader)))

    # save .pth every 50 epochs
    if epoch % 50 == 49:
        if model_name == 'srcnn':
            torch.save(model.state_dict(), r'weights/srcnn/srcnn_'+str(epoch+1)+'.pth')
        elif model_name == 'attentionzbw':
            torch.save(model.state_dict(), r'weights/attentionzbw/attentionzbw_'+str(epoch+1)+'.pth')
               
print('Successful Training!')

