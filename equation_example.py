'''

Уравнение гармонического осцилятора
d^2x/d^2t + w^2 * x = 0
w = 2 * pi * nu

Граничные условия
x0_bc = 1
dx_dt0_bc = 0

'''

import torch
import numpy as np
from tqdm import tqdm
import neural_network

def lossPDE(t, nu = 2):
    
    #вычисляю выходное значение
    out_x = model(t).to(device)
    
    #для упращения
    x = out_x
    
    #вычисляю диф уравнение
    w = 2 * torch.pi * nu
    dx_dt = torch.autograd.grad(x, t, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    d2x_dwt = torch.autograd.grad(dx_dt, t, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    f = dx_dt + d2x_dwt + (w**2) * x
    
    
    loss(f, torch.zeros_like(f))
    
    return loss

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
    
    loss = metric_data(x0, x0_bc) + metric_data(dx0_dt0, dx_dt0_bc)
    
    return loss
    

def loss(input_t):
    loss_bc = lossBC(input_t)
    loss_pde = lossPDE(input_t, 2)
    
    loss = 1e3*loss_bc + loss_pde
    
    return loss

def train(steps = 100):

    model = init_nn();

    pbar = tqdm(range(steps), desc='Training Progress')
    input_t = (torch.linspace(0, 1, 100).unsqueeze(1)).to(device)
    input_t.requires_grad = True

    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)

    for step in pbar:



    
    