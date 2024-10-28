'''

Уравнение гармонического осцилятора
d^2x/d^2t + w^2 * x = 0
w = 2 * pi * nu

Граничные условия
x0_bc = 1
dx_dt0_bc = 0

'''

import neural_network as net
import matplotlib.pyplot as plt
from neural_network import torch
from neural_network import device

# torch.manual_seed(44)
# np.random.seed(44)
# random.seed(44)

def lossBC(t):
    x0_true=torch.tensor([1], dtype=float).float().to(device)
    dx0dt_true=torch.tensor([0], dtype=float).float().to(device)
    inlet_mask = (t[:, 0] == 0)
    t0 = t[inlet_mask]
    x0 = model(t0).to(device)
    dx0dt = torch.autograd.grad(x0, t0, torch.ones_like(t0), create_graph=True, \
                        retain_graph=True)[0]

    loss_bс = model.loss(x0, x0_true) + \
                model.loss(dx0dt, dx0dt_true.to(device))
    return loss_bс

def lossPDE(t, nu = 2):
    out = model(t).to(device)

    omega = 2 * torch.pi * nu
    dxdt = torch.autograd.grad(out, t, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    d2xdt2 = torch.autograd.grad(dxdt, t, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    f1 = d2xdt2 + (omega ** 2) * out
    
    loss_pde = model.loss(f1, torch.zeros_like(f1))
    
    return loss_pde

def fullLoss(t):
    loss = lossPDE(t) + 1e3*lossBC(t)
    return loss

def print_value():
    nu=2
    omega = 2 * torch.pi * nu
    x0_true=torch.tensor([1], dtype=float).float().to(device)
    dx0dt_true=torch.tensor([0], dtype=float).float().to(device)

    t = torch.linspace(0, 2, 100).unsqueeze(-1).unsqueeze(0).to(device)
    t.requires_grad=True
    x_pred = model(t.float())
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

model = net.simpleModel(1, 1, 20, 500, loss_func=fullLoss, lr=0.1).to(device)


t = (torch.linspace(0, 1, 100).unsqueeze(1)).to(device)
t.requires_grad = True

model.training_a(t)

print_value()
