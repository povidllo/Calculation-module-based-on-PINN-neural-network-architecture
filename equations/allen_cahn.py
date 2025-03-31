import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mongo_schemas import *
from mNeural_abs import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
import jinja2
import io
import base64
import abc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from torchviz import make_dot

from torchvision import models
from torchsummary import summary
import hiddenlayer as hl

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

from optim_Adam_torch import create_optim
from pinn_init_torch import pinn
from test_data_generator import generator as test_data_generator

from cfg_main import get_config
torch.manual_seed(123)
np.random.seed(44)
torch.cuda.manual_seed(44)
# print(get_config().epochs)
# -----------------

import torch
from pprint import pprint
import numpy as np

from Modules.pinn_init_torch import pinn
from Modules.optim_Adam_torch import create_optim
from Modules.train_torch import Train_torch
from Modules.allen_cahn.data_generator import data_generator
from Modules.allen_cahn.loss_calc import loss_calculator
from Modules.allen_cahn.calculate_l2 import calculate_l2_error
from Modules.allen_cahn.vizualizer import vizualize
from Modules.allen_cahn.test_data_generator import generator as test_data_generator
# import cfg_pinn_init as cfg_pinn_init
import cfg_main as cfg_main
# import cfg_train_torch as cfg_train_torch

torch.manual_seed(123)
# np.random.seed(44)
# torch.cuda.manual_seed(44)

model = pinn(cfg_main.get_config())
# Вывод весов при инициализации
# for name, param in model.named_parameters():
#     print(f"\nLayer: {name}")
#     print(f"Shape: {param.shape}")
#     print(f"Values:\n{param.data}")
# exit()

optimizer = create_optim(model, cfg_main.get_config())

trainer = Train_torch(cfg_main.get_config(),
                    model,
                    optimizer,
                    data_generator,
                    loss_calculator,
                    test_data_generator,
                    calculate_l2_error,
                    vizualize)
trainer.train()
# -----------------

class oscillator_nn(abs_neural_net):
    mymodel = None
    mydevice = None
    myoptimizer = None

    loss_history = []
    l2_history = []
    best_loss = float('inf')
    best_epoch = 0

    class mySpecialDataSet(mDataSet_mongo):
        def equation(self, u, tx):
            u_tx = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
            u_t = u_tx[:, 0:1]
            u_x = u_tx[:, 1:2]
            u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
            e = u_t - 0.0001 * u_xx + 5 * u ** 3 - 5 * u
            return e

        def loss_calculator(self, u_pred_f, x_physics, u_pred, y_data):
            physics = self.equation(u_pred_f, x_physics)
            loss2 = torch.mean(physics ** 2)

            loss_u = torch.mean((u_pred - y_data) ** 2)  # use mean squared error
            loss = loss_u + loss2

            return loss
