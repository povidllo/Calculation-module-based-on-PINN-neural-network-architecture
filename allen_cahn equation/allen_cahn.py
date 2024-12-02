import neural_network as net
import matplotlib.pyplot as plt
from neural_network import torch, nn
from neural_network import device
import numpy as np
import scipy.io
import neural_network as net
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import rff.layers
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import rff
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

'''
    Само уравнение
    du_dt - ε * d2u_d2x + λ * u(t,x)^3 - λ * u = 0

    u(0,x) = x^2 * cos(πx)
    u(t,-1) = u(t,1)
    du_dx(t,-1) = du_dx(t,1)


    например
    ε = 0.0001
    λ = 5
    t ∊ [0,1]
    x ∊ [-1,1]
'''

'''
    Составляем данные для обучения
'''

t_left_border = 0
t_right_border = 1
t_count = 100
t = torch.linspace(t_left_border, t_right_border, t_count).reshape(-1, 1)

x_left_border = -1
x_right_border = 1
x_count = 256
x = torch.linspace(x_left_border, x_right_border, x_count).reshape(-1, 1)


t_in = np.tile(t, (1, x_count)).T.flatten()[:, None]
x_in = np.tile(x, (1, t_count)).flatten()[:, None]

#реализовать рандомную выборку
#убедиться с тем, как идет (x,t) или (t,x)

input = torch.tensor(np.concatenate((t_in, x_in),1), dtype=torch.float32)
input.requires_grad=True
'''
    Реализация функций потерь
'''
#реализовать для граничных условий
def lossPDE(input,ε=0.0001,λ=5):
    u = model(input)
    du = torch.autograd.grad(u, input, torch.ones_like(u), create_graph=True, \
                                retain_graph=True)[0]
    du_dt = du[:,0]
    du_dx = du[:,1]
    d2u_d2x = torch.autograd.grad(du_dx, input, torch.ones_like(du_dx), create_graph=True, \
                                retain_graph=True)[0][:,1]
    f = du_dt - ε * d2u_d2x + λ * u**3 - λ * u
    return torch.mean(f**2)
    
def lossIC(input):
    initial_mask = input[:, 0] == 0
    initial_points = input[initial_mask]
    x_initial = initial_points[:, 1]  # Координаты x при t=0
    u_initial_pred = model(initial_points)  # Предсказание модели для начального условия
    u_initial_true = x_initial**2 * torch.cos(np.pi * x_initial)  # Истинное значение
    loss_ic = torch.mean((u_initial_pred - u_initial_true)**2)  # Потеря для начального условия
    return loss_ic

'''
    Реализация нейронной сети
'''
#надо посмотреть про инициализацию слоев, какую использовать и надо ли
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AllenCahnModel(nn.Module):
    def __init__(self,
                 input_size=2,
                 hidden_size=256,
                 output_size=1):
        super().__init__()

        self.layers_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #1
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #2
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #3
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #4
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        return self.layers_stack(x)

model = AllenCahnModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
steps = 1000
pbar = tqdm(range(steps), desc='Training Progress')
writer = SummaryWriter()

for step in pbar:
    def closure():
        optimizer.zero_grad()

        loss_ic = lossIC(input)
        loss_pde = lossPDE(input)
        loss = loss_ic + loss_pde
        
        loss.backward()
        return loss

    optimizer.step(closure)
    if step % 2 == 0:
        current_loss = closure().item()
        pbar.set_description("Step: %d | Loss: %.6f | " %
                                (step, current_loss))
        writer.add_scalar('Loss/train', current_loss, step)
        
writer.close()
