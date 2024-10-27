'''

Уравнение гармонического осцилятора
d^2x/d^2t + w^2 * x = 0
w = 2 * pi * nu

Граничные условия
x0_bc = 1
dx_dt0_bc = 0

'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import neural_network as nene

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def lossPDE(t, nu = 2):
    
    #вычисляю выходное значение
    out_x = model(t).to(device)
    
    #для упрощения
    x = out_x
    
    #вычисляю диф уравнение
    w = 2 * torch.pi * nu
    dx_dt = torch.autograd.grad(x, t, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    d2x_dwt = torch.autograd.grad(dx_dt, t, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    f = dx_dt + d2x_dwt + (w**2) * x
    
    
    lossPDE = model.loss(f, torch.zeros_like(f))
    
    return lossPDE

def lossBC(input_t):
    #устанавливаю граничные условия
    x0_bc = torch.tensor([1], dtype=float).float().to(device)
    dx_dt0_bc = torch.tensor([0], dtype=float).float().to(device)
    
    #маска, где true только на 0 значении
    mask = (input_t[:,0] == 0)
    
    t0 = input_t[mask] # = 0.000
    
    #вычисляем x0 на граничных условиях и производную
    x0 = model(t0).to(device)
    dx0_dt0 = torch.autograd.grad(x0, t0, torch.ones_like(t0), create_graph=True, retain_graph=True)[0]
    
    loss = model.loss(x0, x0_bc) + model.loss(dx0_dt0, dx_dt0_bc)
    
    return loss
    

def complete_loss(input_t):
    loss_bc = lossBC(input_t)
    loss_pde = lossPDE(input_t, 2)
    
    loss = 1e3*loss_bc + loss_pde
    
    return loss

def train(epochs = 100):
    input_t = (torch.linspace(0, 1, 100).unsqueeze(1)).to(device)
    input_t.requires_grad = True

    losses = model.fit(input_t)
    

model = nene.init_nn(1, 1, 20, 500, loss1=lossBC, loss2=lossPDE).to(device)

train()


t = torch.linspace(0, 1, 100).unsqueeze(-1).unsqueeze(0).to(device)
t.requires_grad=True
x_pred = model(t.float())    

x0_true = torch.tensor([1], dtype=float).float().to(device)
nu=2
omega = 2 * torch.pi * nu

x_true = x0_true * torch.cos(omega*t)

fs=13
plt.scatter(t[0].cpu().detach().numpy(), x_pred[0].cpu().detach().numpy(), label='pred',
            marker='o',
            alpha=.7,
            s=50)
plt.plot(t[0].cpu().detach().numpy(),x_true[0].cpu().detach().numpy(),
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