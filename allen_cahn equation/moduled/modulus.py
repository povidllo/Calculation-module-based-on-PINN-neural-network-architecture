import torch
from torch import nn

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


class PINN(nn.Module):
    def __init__(self, hidden_size):
      super().__init__()
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      print(2)
      self.layers_stack = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(2, hidden_size), dim = 0),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size), dim=0),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size), dim=0),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size), dim=0),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, x):
      return self.layers_stack(x)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def pinn(hidden_size):
    model = PINN(hidden_size)
    model.apply(weights_init)
    return model