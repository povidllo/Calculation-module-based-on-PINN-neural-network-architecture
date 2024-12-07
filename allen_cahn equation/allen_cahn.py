import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt

"""###Сохранение результатов"""

# !pip install wandb -qU

# import wandb
# import random
# import math

# wandb.login()

# wandb.init(
#     # Set the project where this run will be logged
#     project="Allen-cahn",
#     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#     name=f"переделанный",
#     # Track hyperparameters and run metadata
#     config={
#     "epochs": 100000,
#     })

"""###Импортированный блок

data.py
"""

import numpy as np

def ac_generator(num_t, num_x, typ='train'):
    N_f = num_t*num_x
    t = np.linspace(0, 1, num_t).reshape(-1,1) # T x 1
    x = np.linspace(-1, 1, num_x).reshape(-1,1) # N x 1
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T

    # Initial condition and boundary condition
    u = np.zeros((N, T))  # N x T
    u[:,0:1] = (x**2)*np.cos(np.pi*x)
    u[0,:] = -np.ones(T)
    u[-1,:] = u[0,:]

    t_data = T_star.flatten()[:, None]
    x_data = X_star.flatten()[:, None]
    u_data = u.flatten()[:, None]

    t_data_f = t_data.copy()
    x_data_f = x_data.copy()

    if typ == 'train':
        idx = np.random.choice(np.where((x_data == -1) | (x_data == 1))[0], num_t)
        t_data = t_data[idx]
        x_data = x_data[idx]
        u_data = u_data[idx]

        init_idx = np.random.choice(N-1, num_x-4, replace=False) + 1
        t_data = np.concatenate([t_data, np.ones((2,1)), np.zeros((num_x-4,1))], axis=0)
        x_data = np.concatenate([x_data, np.array([[-1], [1]]), x[init_idx]], axis=0)
        u_data = np.concatenate([u_data, -np.ones((2,1)), u[init_idx,0:1]], axis=0)

        return t_data, x_data, u_data, t_data_f, x_data_f

    else:
        return t_data_f, x_data_f

"""    model.py"""

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

"""      utils.py"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0]
    return derivative


def ac_equation(u, tx):
    # u_tx = fwd_gradients(u, tx)
    u_tx = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph= True)[0]
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    # u_xx = fwd_gradients(u_x, tx)[:, 1:2]
    u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph= True)[0][:, 1:2]
    e = u_t -0.0001*u_xx + 5*u**3 - 5*u
    return e

def resplot(x, t, t_data, x_data, Exact, u_pred):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x, Exact[:,0],'-')
    plt.plot(x, u_pred[:,0],'--')
    plt.legend(['Reference', 'Prediction'])
    plt.title("Initial condition ($t=0$)")

    plt.subplot(2, 2, 2)
    t_step = int(0.25*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    plt.legend(['Reference', 'Prediction'])
    plt.title("$t=0.25$")

    plt.subplot(2, 2, 3)
    t_step = int(0.5*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    plt.legend(['Reference', 'Prediction'])
    plt.title("$t=0.5$")

    plt.subplot(2, 2, 4)
    t_step = int(0.99*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    plt.legend(['Reference', 'Prediction'])
    plt.title("$t=0.99$")
    plt.show()
    plt.close()

"""### 1. Default Setting

1. Domain: 100 x 256 ($x \in [-1,1]$ and $t \in [0,1]$)

2. Collocation points: $N_{ic}=256$ and $N_{f}=25600$

3. Optimizer: Adam with the learning rate of $10^{-3}$

"""

torch.manual_seed(44)
np.random.seed(44)

num_t = 100
num_x = 256
num_epochs = 100000
num_hidden = 4
num_nodes = 128
lr = 1e-3

# Select a partial differential equation
eq = 'ac' # or 'bg'

"""### 2. Train Data"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Operation mode: ", device)

if eq == 'ac':
    t_data, x_data, u_data, t_data_f, x_data_f = ac_generator(num_t, num_x)
else:
    print("There exists no the equation.")
    exit(0)

variables = torch.FloatTensor(np.concatenate((t_data, x_data), 1)).to(device)
variables_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), 1)).to(device)
variables_f.requires_grad = True
u_data = torch.FloatTensor(u_data).to(device)

"""### 3. Neural Network"""

# layer_list = [2] + num_hidden * [num_nodes] + [1]
# pinn = pinn(layer_list).to(device)
pinn = pinn(num_nodes).to(device)

"""### 4. Training Session"""

optimizer = torch.optim.Adam(pinn.parameters(), betas=(0.999,0.999), lr=lr)

loss_graph = []
ls = 1e-3
bep = 0

def calculateL2():
  t = np.linspace(0, 1, 201).reshape(-1,1) # T x 1
  x = np.linspace(-1, 1, 513)[:-1].reshape(-1,1) # N x 1
  T = t.shape[0]
  N = x.shape[0]
  T_star = np.tile(t, (1, N)).T  # N x T
  X_star = np.tile(x, (1, T))  # N x T
  t_test = T_star.flatten()[:, None]
  x_test = X_star.flatten()[:, None]

  test_variables = torch.FloatTensor(np.concatenate((t_test, x_test), 1)).to(device)
  with torch.no_grad():
      u_pred = pinn(test_variables)
  u_pred = u_pred.cpu().numpy().reshape(N,T)

  # reference data
  data = scipy.io.loadmat('./data/AC.mat')
  Exact = np.real(data['uu'])
  err = u_pred-Exact
  err = np.linalg.norm(err,2)/np.linalg.norm(Exact,2)
  # print(f"L2 Relative Error: {err}")
  return err

for ep in tqdm(range(num_epochs)):

        optimizer.zero_grad()

        # Full batch
        u_hat = pinn(variables)
        u_hat_f = pinn(variables_f)

        loss_f = torch.mean(ac_equation(u_hat_f, variables_f) ** 2)

        loss_u = torch.mean((u_hat - u_data) ** 2)
        loss = loss_f + loss_u
        loss.backward()
        optimizer.step()

        l = loss.item()
        loss_graph.append(l)
        if l < ls:
            ls = l
            bep = ep
            torch.save(pinn.state_dict(), './'+eq+'_1d.pth')

        if ep % 100 == 0:
            print(f"Train loss: {l}")
            # wandb.log({"epoche": ep, "loss": loss})
        if ep % 500 == 0:
            l2 = calculateL2()
            # wandb.log({"epoche": ep, "L2": l2})
            print(f"L2 Relative Error: {l2}")
# wandb.finish()

print(f"[Best][Epoch: {bep}] Train loss: {ls}")
plt.figure(figsize=(10, 5))
plt.plot(loss_graph)
plt.show()

"""### 5. Inference Session"""

pinn.load_state_dict(torch.load('./'+eq+'_1d.pth'))

if eq == 'ac':
    t = np.linspace(0, 1, 201).reshape(-1,1) # T x 1
    x = np.linspace(-1, 1, 513)[:-1].reshape(-1,1) # N x 1
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T
    t_test = T_star.flatten()[:, None]
    x_test = X_star.flatten()[:, None]

    test_variables = torch.FloatTensor(np.concatenate((t_test, x_test), 1)).to(device)
    with torch.no_grad():
        u_pred = pinn(test_variables)
    u_pred = u_pred.cpu().numpy().reshape(N,T)

    # reference data
    data = scipy.io.loadmat('./data/AC.mat')
    Exact = np.real(data['uu'])
    err = u_pred-Exact

err = np.linalg.norm(err,2)/np.linalg.norm(Exact,2)
print(f"L2 Relative Error: {err}")

"""### 6. Result Figures"""

resplot(x, t, t_data, x_data, Exact, u_pred)

plt.figure(figsize=(10, 5))
plt.imshow(u_pred, interpolation='nearest', cmap='jet',
            extent=[t.min(), t.max(), x.min(), x.max()],
            origin='lower', aspect='auto')
plt.clim(-1, 1)
plt.ylim(-1,1)
plt.xlim(0,1)
plt.scatter(t_data, x_data)
plt.xlabel('t')
plt.ylabel('x')
plt.title('u(t,x)')
plt.show()

