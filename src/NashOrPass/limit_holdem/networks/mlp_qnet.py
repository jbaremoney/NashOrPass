import torch.nn as nn


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int = 72, hidden_dim: int = 256, num_actions: int = 4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)