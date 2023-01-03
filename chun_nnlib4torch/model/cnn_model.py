import torch
from torch import nn
import torch.nn.functional as F

class TestConvNet(nn.Module):
    def __init__(self):
        super(TestConvNet, self).__init__()
        # image shape is 1 * 28 * 28, where 1 is one color channel
        # 28 * 28 is the image size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)    # output shape = 3 * 24 * 24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                       # output shape = 3 * 12 * 12
        # intput shape is 3 * 12 * 12
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5)    # output shape = 9 * 8 * 8
        # add another max pooling, output shape = 9 * 4 * 4
        #self.fc1 = nn.Linear(9*4*4, 100)
        #self.fc1 = nn.Linear(1521, 100)
        self.fc1 = nn.Linear(144, 100)
        self.fc2 = nn.Linear(100, 50)
        # last fully connected layer output should be same as classes
        self.fc3 = nn.Linear(50, 10)

        
    def forward(self, x):
        # first conv
        x = self.pool(F.relu(self.conv1(x)))
        # second conv
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        #print(x.view(x.size(0), -1).shape)
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

