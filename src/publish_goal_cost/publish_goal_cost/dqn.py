import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size=95, n_actions=3):
        super(DQN, self).__init__()
        self.inner_dim = 64 * 10 * 10 # Depends on input size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(self.inner_dim, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # Add a channel dimension so we can put our [BATCH, DIM, DIM] 2D map through a convnet
        x = x.unsqueeze(1)
        # x.size() = torch.Size([B, 1, 95, 95])
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)
        # x.size() = torch.Size([B, 32, 46, 46])
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.relu(x)
        # x.size() = torch.Size([B, 64, 22, 22])
        x = self.conv3(x)
        x = self.pool3(x)
        x = torch.relu(x)
        # x.size() = torch.Size([B, 64, 10, 10])
        x = x.view(-1, self.inner_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x