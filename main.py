import torch
import numpy
import pandas
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DebrisDataset

# this determines what does the heavy lifting if you have a nvida GPU with cuda cores or if you dont
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters

learning_rate = 1e-3
batch_size = 64
num_epochs = 1
# this just creates a directory if one doesnt exist
start_dir = "start"
if not os.path.exists(start_dir):
    os.makedirs(start_dir)

data_dir = 'data'
dataset = DebrisDataset(csv_file = "_p.fits", root_dir = "I forgot", transform = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [2000, 1000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle = True)

# Model (what will i do here)
model = torchvision.models
model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss
optimizer = torch.optim.Adam(model.paramaters(), lr = learning_rate)

#training
for epoch in range(num_epochs):
    losses = []
    for batch_index, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # foward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        #backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')


