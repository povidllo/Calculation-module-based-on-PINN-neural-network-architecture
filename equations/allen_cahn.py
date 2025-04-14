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
import scipy.io

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

class allen_cahn_nn(AbsNeuralNet):
    mymodel = None
    mydevice = None
    myoptimizer = None

    loss_history = []
    l2_history = []
    best_loss = float('inf')
    best_epoch = 0


    class mySpecialDataSet(mDataSetMongo):
        def __prepare_data(self):
            '''
            Загрузка и подготовка данных через внешний модуль
            main - данные для обучения физического аспекта модели
            secondary - данные для обучения, путем сравнения с правильными данными
            secondary_true - правильные данные для secondary
            '''
            data = self.data_generator()
            self.variables_f = data['main'].to(self.device)
            self.u_data = data['secondary_true'].to(self.device)
            self.variables = data['secondary'].to(self.device)

        def equation(self, u, tx):
            u_tx = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
            u_t = u_tx[:, 0:1]
            u_x = u_tx[:, 1:2]
            u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
            e = u_t - 0.0001 * u_xx + 5 * u ** 3 - 5 * u
            return e

        def ac_generator(self, num_t, num_x, typ='train'):
            N_f = num_t * num_x
            t = np.linspace(0, 1, num_t).reshape(-1, 1)  # T x 1
            x = np.linspace(-1, 1, num_x).reshape(-1, 1)  # N x 1
            T = t.shape[0]
            N = x.shape[0]
            T_star = np.tile(t, (1, N)).T  # N x T
            X_star = np.tile(x, (1, T))  # N x T

            # Initial condition and boundary condition
            u = np.zeros((N, T))  # N x T
            u[:, 0:1] = (x ** 2) * np.cos(np.pi * x)
            u[0, :] = -np.ones(T)
            u[-1, :] = u[0, :]

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

                init_idx = np.random.choice(N - 1, num_x - 4, replace=False) + 1
                t_data = np.concatenate([t_data, np.ones((2, 1)), np.zeros((num_x - 4, 1))], axis=0)
                x_data = np.concatenate([x_data, np.array([[-1], [1]]), x[init_idx]], axis=0)
                u_data = np.concatenate([u_data, -np.ones((2, 1)), u[init_idx, 0:1]], axis=0)

                return t_data, x_data, u_data, t_data_f, x_data_f

            else:
                return t_data_f, x_data_f

        def data_generator(self):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            t_data, x_data, u_data, t_data_f, x_data_f = self.ac_generator(
                self.num_dots["train"],
                self.num_dots["physics"]
            )
            variables = torch.FloatTensor(np.concatenate((t_data, x_data), axis=1)).to(device)
            variables_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), axis=1)).to(device)
            variables_f.requires_grad = True
            u_data = torch.FloatTensor(u_data).to(device)
            # return {'train': variables_f, 'boundary': variables, 'boundary_true': u_data}
            return {'main': variables_f, 'secondary': variables, 'secondary_true': u_data}

        def loss_calculator(self, u_pred_f, x_physics, u_pred, y_data):
            physics = self.equation(u_pred_f, x_physics)
            loss2 = torch.mean(physics ** 2)

            loss_u = torch.mean((u_pred - y_data) ** 2)  # use mean squared error
            loss = loss_u + loss2

            return loss

        def calculate_l2_error(self, path_true_data, model, device, test_data_generator):
            test_data, _, [N, T] = test_data_generator()
            test_variables = torch.FloatTensor(test_data).to(device)
            with torch.no_grad():
                u_pred = model(test_variables)
            u_pred = u_pred.cpu().numpy().reshape(N, T)

            # Сравнение с эталоном
            data = scipy.io.loadmat(sys.path[0] + path_true_data)
            exact_solution = np.real(data['uu'])
            error = np.linalg.norm(u_pred - exact_solution, 2) / np.linalg.norm(exact_solution, 2)
            return error


    async def set_dataset(self):
        if not self.neural_model.data_set:
            # Создаем новый датасет
            dataset = self.mySpecialDataSet(
                num_dots={
                    "train": 201,
                    "physics": 513
                }
            )
            print('Creating new dataset')
            data = dataset.data_generator()
            # Сохраняем датасет в базу
            await dataset.insert()

            # Обновляем ссылку на датасет в модели
            self.neural_model.data_set = [dataset]
            await mNeuralNetMongo.m_save(self.neural_model)
        else:
            print('Loading existing dataset')
            # Получаем данные из существующего датасета
            existing_data = self.neural_model.data_set[0]

            # Создаем новый экземпляр mySpecialDataSet с теми же данными
            dataset = self.mySpecialDataSet(**existing_data.model_dump())

            # Заменяем старый датасет новым в модели
            self.neural_model.data_set = [dataset]
            data = dataset.data_generator()

        self.variables_f = data['main'].to(self.mydevice)
        self.u_data = data['secondary_true'].to(self.mydevice)
        self.variables = data['secondary'].to(self.mydevice)

    async def set_optimizer(self, opti = None):
        if opti is None:
            if self.neural_model.optimizer:
                opti = self.neural_model.optimizer[0]
                await opti.save()
            else:
                await self.abs_set_optimizer()
                opti = self.neural_model.optimizer[0]
                print('Создан новый оптимизатор:', opti)

        # Создаем PyTorch оптимизатор в зависимости от метода
        if opti.method == 'Adam':
            print('Используем Adam')
            self.torch_optimizer = torch.optim.Adam(
                self.mymodel.parameters(),
                **opti.params
            )
        elif opti.method == 'SGD':
            print('Используем SGD')
            self.torch_optimizer = torch.optim.SGD(
                self.mymodel.parameters(),
                **opti.params
            )
        elif opti.method == 'RMSprop':
            print('Используем RMSprop')
            self.torch_optimizer = torch.optim.RMSprop(
                self.mymodel.parameters(),
                **opti.params
            )
        elif opti.method == 'Adagrad':
            print('Используем Adagrad')
            self.torch_optimizer = torch.optim.Adagrad(
                self.mymodel.parameters(),
                **opti.params
            )
        else:
            raise ValueError(f"Неизвестный метод оптимизации: {opti.method}")

    async def construct_model(self, params, in_device):
        await self.create_model(params)

        self.mydevice = in_device
        self.mymodel = pinn(params).to(self.mydevice)
        await self.set_optimizer()

    async def save_weights(self, path):
        print('run saving')
        weights = self.mymodel.state_dict()
        await self.abs_save_weights(weights)
        print('save complited')

    async def train(self):
        self.config = self.neural_model.hyper_param
        epochs = self.config.epochs

