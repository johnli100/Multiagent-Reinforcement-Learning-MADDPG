import torch
from torch import nn

class Network_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[48, 24]):
        super().__init__()
        hidden_dims = [state_dim] + hidden_dims
        self.layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) \
                                      for i in range(len(hidden_dims)-1)])
        self.layer_output = nn.Linear(hidden_dims[-1], action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return self.tanh(self.layer_output(x))


class Network_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 32]):
        super().__init__()
        hidden_dims = [state_dim+action_dim] + hidden_dims
        self.layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) \
                                      for i in range(len(hidden_dims)-1)])
        self.layer_output = nn.Linear(hidden_dims[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return self.layer_output(x)