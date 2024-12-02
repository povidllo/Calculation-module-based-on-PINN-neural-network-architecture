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

# def np_to_th(x):
#     n_samples = len(x)
#     return torch.from_numpy(x).to(torch.float).to(device).reshape(n_samples, -1)

# def func(x, x0, omega, nu):
#     x_true = x0 * np.cos(omega*x)
#     return x_true

# def ret():
#     nu=2
#     omega = 2 * torch.pi * nu
#     times = np.linspace(0, 2, 100)
#     eq = functools.partial(func,x0=1,omega=omega,nu=nu)
#     temps = eq(times)

#     x = np.linspace(0, 3, 30)
#     y = eq(x) +  0.8 *np.random.rand()
#     return x, y

def fourier_feature(x, in_dim, map):
        freqs = nn.Parameter(torch.randn(map, in_dim))  # Размерность частот
        scale = nn.Parameter(torch.randn(map))  # Масштабирование
        sinusoid = torch.cat([torch.sin(x @ freqs.T + scale), torch.cos(x @ freqs.T + scale)], dim=-1)
        return sinusoid


class simpleModel(nn.Module):
    def __init__(self,
                true,
                input_size=1,
                output_size=1,
                hidden_size=12,
                epoch=900,
                loss=nn.MSELoss(),
                loss_func=None,  # <- обновлено, чтобы loss_func работала корректно
                lr=0.01,
                fourie=False,
                mapped_fourie=256,
                ):
        super().__init__()
        self.true = true
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.epoch = epoch
        self.loss = loss
        self.loss_func = loss_func  # <- сохраняем loss_func для использования
        self.lr = lr
        self.fourie=fourie
        self.mapped_fourie = mapped_fourie

        self.layers_stack = nn.Sequential(
            nn.Linear(2*mapped_fourie if fourie else input_size, hidden_size),
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
        if(self.fourie):
            # encoding = rff.layers.PositionalEncoding(sigma=1.0, m=self.mapped_fourie)
            # encoding = rff.layers.BasicEncoding()
            encoding = rff.layers.GaussianEncoding(sigma=5.0, input_size=self.input_size, encoded_size=self.mapped_fourie)         
            xf = encoding(x)
            return self.layers_stack(xf)
            # xf = fourier_feature(x, self.input_size, self.mapped_fourie)
            # return self.layers_stack(xf)
        return self.layers_stack(x)
    
    def myl2Loss(self, input, true):
        out = self.forward(input)
        
        true_new = torch.tensor(true.flatten()).unsqueeze(-1)
        diff = out - true_new
        l2_norm = torch.sqrt(torch.mean(diff**2))
        return l2_norm.item()
    
    def training_a(self, t):
        # X, y = ret()

        # Xt = np_to_th(X)
        # yt = np_to_th(y)

        steps = self.epoch
        pbar = tqdm(range(steps), desc='Training Progress')
        writer = SummaryWriter()
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        for step in pbar:
            def closure():
                optimizer.zero_grad()
                # outputs = self.forward(Xt)
                # loss = self.loss(yt, outputs)
                # loss += self.loss_func(t)  # <- используем loss_func корректно
                loss = self.loss_func(t, self.true)
                loss.backward()
                return loss

            optimizer.step(closure)
            if step % 2 == 0:
                current_loss = closure().item()
                l2 = self.myl2Loss(t, self.true)
                pbar.set_description("Step: %d | Loss: %.6f | L2: %.6f" %
                                     (step, current_loss, l2))
                writer.add_scalar('Loss/train', current_loss, step)
                writer.add_scalar('L2/train', l2, step)
        writer.close()




