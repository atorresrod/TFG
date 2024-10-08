import torch
import torch.nn as nn

class DQN(nn.Module):
    
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

