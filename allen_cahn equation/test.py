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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
du_dt - ε * d2u_d2x + λ * u(t,x)^3 - λ * u = 0

u(0,x) = x^2 * cos(πx)
u(t,-1) = u(t,1)
du_dx(t,-1) = du_dx(t,1)

ε = 0.0001
λ = 5
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка данных
def get_dataset():
    data = scipy.io.loadmat("./allen_cahn equation/dataset/allen_cahn.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    return u_ref, t_star, x_star

def print_results():
    u_ref, t_star, x_star = get_dataset()
    T, X = np.meshgrid(t_star, x_star)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, u_ref.T, cmap='jet')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_zlabel('u(t, x)')
    ax.set_title('Allen-Cahn Equation Solution')
    
    plt.tight_layout()
    plt.show()

def lossPDE(input, epsilon=0.0001, lamb=5):

    u = model(input).to(device)
    du_dt = torch.autograd.grad(u, u[:,0], torch.ones_like(u[:,0]), create_graph=True, \
                                retain_graph=True)[0]
    du_dx = torch.autograd.grad(u, u[:,1], torch.ones_like(u[:,1]), create_graph=True, \
                            retain_graph=True)[0]
    d2udx2 = torch.autograd.grad(du_dx, u[:,1], torch.ones_like(du_dx), create_graph=True, \
                            retain_graph=True)[0]
    # du_dt - ε * d2u_d2x + λ * u(x,t)^3 - λ * u = 0
    f1 = du_dt - epsilon*d2udx2 + lamb*u**3 - lamb * u
    loss_pde = torch.mean(f1**2)
    
    return loss_pde




def lossIC(input, true_out):
    inlet_mask = (t_tensor == 0)  # Маска для t == 0
    input_t0 = input_data[inlet_mask.repeat(len(x_tensor), 1).view(-1)]  # Применяем маску
    out_IC = torch.tensor(out_dataset[0, :], dtype=torch.float32).to(device)
    # print(inlet_mask)
    # print("input data с маской на 0:", input_t0)
    out_nn = model(input_t0)
    # loss_ic = model.loss(out_IC, out_nn)
    loss_ic = torch.mean(out_IC - out_nn)**2
    return loss_ic


out_dataset, t, x = get_dataset()
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
t_tensor = torch.tensor(t, dtype=torch.float32).to(device)
out_tensor = torch.tensor(out_dataset, dtype=torch.float32).to(device)
x_all_t = x_tensor.repeat(len(t_tensor), 1).T.to(device)
t_all_x = t_tensor.repeat(len(x_tensor), 1).to(device)

prepared_data = torch.stack((t_all_x, x_all_t), dim=2).to(device)  #(t, x)


# Преобразуем в одномерный массив для подачи в модель
# Каждый элемент будет иметь форму (x, t) для подачи в модель
# Сначала делаем данные одномерными
input_data = prepared_data.view(-1, 2).to(device)  # Преобразуем в форму (len(x_star) * len(t_star), 2)
input_data.requires_grad=True
# Преобразование в нужный формат для модели (например, [batch_size, input_size])
# print("input data:", input_data)



class simpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers_stack = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 256), #1
            nn.Tanh(),
            nn.Linear(256, 256), #2
            nn.Tanh(),
            nn.Linear(256, 256), #3
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.layers_stack(x)
    

def myl2Loss(input, true):
    out = model(input)
    
    true_new = true.flatten().unsqueeze(-1)
    diff = out - true_new
    l2_norm = torch.sqrt(torch.mean(diff**2))
    return l2_norm.item()

model = simpleModel()
steps = 1000
pbar = tqdm(range(steps), desc='Training Progress')
writer = SummaryWriter()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
for step in pbar:
    def closure():
        optimizer.zero_grad()

        loss_ic = lossIC(input_data, out_tensor)
        loss_pde = lossPDE(input_data)
        loss = loss_ic + loss_pde
        
        loss.backward()
        return loss

    optimizer.step(closure)
    if step % 2 == 0:
        current_loss = closure().item()
        l2 = myl2Loss(input_data, out_tensor)
        pbar.set_description("Step: %d | Loss: %.6f | L2: %.6f" %
                                (step, current_loss, l2))
        writer.add_scalar('Loss/train', current_loss, step)
        writer.add_scalar('L2/train', l2, step)
writer.close()


