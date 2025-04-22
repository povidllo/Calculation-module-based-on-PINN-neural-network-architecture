import torch

def ac_equation(u, tx):
    u_tx = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph= True)[0]
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph= True)[0][:, 1:2]
    e = u_t -0.0001*u_xx + 5*u**3 - 5*u
    return e

def loss_calculator(u_pred_f, variables_f, u_pred, u_data):
    loss_f = torch.mean(ac_equation(u_pred_f, variables_f) ** 2)
    loss_u = torch.mean((u_pred - u_data) ** 2)
    loss = loss_f + loss_u

    return loss     