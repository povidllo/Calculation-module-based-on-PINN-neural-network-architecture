import torch

def equation(yhp, x_physics, d=2, w0=20):
    mu = d * 2
    k = w0 ** 2
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
    dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]
    physics = dx2 + mu*dx + k*yhp
    return physics


def loss_calculator(yhp, x_physics, yh, y_data):
    # loss_f = torch.mean(ac_equation(u_pred_f, variables_f) ** 2)
    # loss_u = torch.mean((u_pred - u_data) ** 2)
    # loss = loss_f + loss_u
    loss1 = torch.mean((yh-y_data)**2)# use mean squared error
    
    physics = equation(yhp, x_physics)
    loss2 = (1e-4)*torch.mean(physics**2)

    loss = loss1 + loss2    

    return loss     