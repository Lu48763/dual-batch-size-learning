#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms


# In[ ]:


root = '~/.pytorch/datasets/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


mean = torch.tensor([129.3, 124.1, 112.4]) / 255
std = torch.tensor([68.2, 65.4, 70.4]) / 255

train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std),
     transforms.RandomCrop(32, padding=4, padding_mode='constant'),
     transforms.RandomHorizontalFlip()])
eval_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])

trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
testset = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)


# In[ ]:


model = models.resnet18()
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Sequential()
model.fc = nn.Linear(in_features=512, out_features=100, bias=True)
model = model.to(device)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=0)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)


# In[ ]:


num_workers = 2


# In[ ]:


start_batch = 1
end_batch = 1000 + 1
step = 1
record_train_time = np.zeros((end_batch - start_batch) // step + 1)


# In[ ]:


counter = 0
for batch_size in range(start_batch, end_batch, step):
    print(f'Batch Size = {batch_size}')
    # start
    record_train_time[counter] = time.time()
    # dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    # loss acc initialization
    train_loss, train_acc = 0, 0
    #train
    model.train()
    for i, data in enumerate(trainloader):
        # load data
        inputs = data[0].to(device)
        labels = data[1].to(device)
        # compute
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # record
        predict_labels = torch.max(outputs, dim=1).indices
        train_loss += loss.item()
        train_acc += torch.sum(labels==predict_labels).item()
    train_loss /= len(trainloader)
    train_acc /= len(trainset)
    print(f'train_loss: {train_loss}, train_acc: {train_acc}')
    # end
    record_train_time[counter] = time.time() - record_train_time[counter]
    counter += 1


# In[ ]:


np.save(f'pt_train_time_{start_batch}_{end_batch}_{step}.npy', record_train_time)


# In[ ]:




