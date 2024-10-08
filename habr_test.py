import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from google.colab import files as filescolab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class simpleModel(nn.Module):
  def __init__(self,
               hidden_size=20):
    super().__init__()
    self.layers_stack = nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #1
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #2
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #3
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #4
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
        nn.Tanh(),
    )

  def forward(self, x):
    return self.layers_stack(x)


def pde(out, t, nu=2):
    omega = 2 * torch.pi * nu
    dxdt = torch.autograd.grad(out, t, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    d2xdt2 = torch.autograd.grad(dxdt, t, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    f = d2xdt2 + (omega ** 2) * out
    return f

# def pde(out, x, C=1):
#     du_dx = torch.autograd.grad(out, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
#     d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
#
#     f = C * x
#
#     residual = d2u_dx2 + f
#     return residual


nu=2
omega = 2 * torch.pi * nu
x0_true=torch.tensor([1], dtype=float).float().to(device)
dx0dt_true=torch.tensor([0], dtype=float).float().to(device)
# C = 1.0
# T_star = 1.0
# u_star = 1.0
# L = 10.0
# x0_true=torch.tensor([-T_star], dtype=float).float().to(device)
# dx0dt_true=torch.tensor([u_star], dtype=float).float().to(device)


model = simpleModel().to(device)

steps=1000
pbar = tqdm(range(steps), desc='Training Progress')
t = (torch.linspace(0, 1, 100).unsqueeze(1)).to(device)
t.requires_grad = True

metric_data = nn.MSELoss()
writer = SummaryWriter()
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)


def pdeBC(t):
    out = model(t).to(device)
    f1 = pde(out, t)

    inlet_mask = (t[:, 0] == 0)
    t0 = t[inlet_mask]
    x0 = model(t0).to(device)
    dx0dt = torch.autograd.grad(x0, t0, torch.ones_like(t0), create_graph=True, \
                        retain_graph=True)[0]

    loss_bc = metric_data(x0, x0_true) + \
                metric_data(dx0dt, dx0dt_true.to(device))
    loss_pde = metric_data(f1, torch.zeros_like(f1))

    loss = 1e3*loss_bc + loss_pde

    # metric_x = metric_data(out, x0_true * torch.sin(omega*t + torch.pi / 2))
    # metric_x0 = metric_data(x0, x0_true)
    # metric_dx0dt = metric_data(dx0dt, dx0dt_true.to(device))

    # acc_metrics = {'metric_x': metric_x,
    #             'metric_x0': metric_x0,
    #             'metric_dx0dt': metric_dx0dt,
    #             }

    # metrics = {'loss': loss,
    #             'loss_bc': loss_bc,
    #             'loss_pde': loss_pde,
    #             }

    return loss




def train():


    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = pdeBC(t)
            loss.backward()
            return loss

        optimizer.step(closure)
        if step % 2 == 0:
            current_loss = closure().item()
            pbar.set_description("Step: %d | Loss: %.6f" %
                                 (step, current_loss))
            writer.add_scalar('Loss/train', current_loss, step)





def predict():
    t = torch.linspace(0, 1, 100).unsqueeze(-1).unsqueeze(0).to(device)
    t.requires_grad = True
    x_pred = model(t.float())

    x_true = x0_true * torch.cos(omega*t)
    # x_true = C/6 *(L**3 - t**3) - T_star*(t+L) + u_star

    fs = 13
    plt.scatter(t[0].cpu().detach().numpy(), x_pred[0].cpu().detach().numpy(), label='pred',
                marker='o',
                alpha=.7,
                s=50)
    plt.plot(t[0].cpu().detach().numpy(), x_true[0].cpu().detach().numpy(),
             color='blue',
             label='analytical')
    plt.xlabel('t', fontsize=fs)
    plt.ylabel('x(t)', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend()
    plt.title('x(t)')
    plt.savefig('x.png')
    plt.show()



if __name__ == '__main__':
    train()
    writer.close()
    predict()