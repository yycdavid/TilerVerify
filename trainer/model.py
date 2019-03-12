import torch
import torch.nn as nn
import torch.nn.functional as F


# In order to be compatible with MIPVerify, the conv layer must have padding=1, and the input to each conv layer must have size that is even

class CNN_small(nn.Module):
    ''' Small CNN model for 32*32 input'''
    def __init__(self):
        super(CNN_small, self).__init__()
        # Input 32*32*1
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        # 16*16*16
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        # 8*8*32
        self.fc1 = nn.Linear(8*8*32, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        # input needs to have shape (batch_size, 1, 32, 32)
        # output (batch_size, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
