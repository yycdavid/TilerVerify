import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_small(nn.Module):
    ''' Small CNN model for 32*32 input'''
    def __init__(self):
        super(Net, self).__init__()
        # Input 32*32*1
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2)
        # 14*14*16
        self.conv2 = nn.Conv2d(16, 32, 4, strid=2)
        # 5*5*32
        self.fc1 = nn.Linear(5*5*32, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        # input needs to have shape (batch_size, 1, 32, 32)
        # output (batch_size, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 5*5*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
