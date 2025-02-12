import jinja2
import io
import base64
from mongo_schemas import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
import abc

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from torchviz import make_dot

from torchvision import models
from torchsummary import summary
import hiddenlayer as hl


class simpleModel(nn.Module):
  def __init__(self,
               hidden_size=20):
    super().__init__()
    self.layers_stack = nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #1
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #2
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #3
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #4
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
        nn.Tanh(),
    )


  def forward(self, x):
    return self.layers_stack(x)


class abs_neural_net(abc.ABC):
    neural_model : mNeuralNet_mongo = None


    async def createModel(self, params : mHyperParams):
        self.neural_model = mNeuralNet_mongo(hyper_param=mHyperParams_mongo(**params.model_dump()))
        self.set_dataset()

        # await mNeuralNet_mongo.m_insert(self.neural_model)


    @abc.abstractmethod
    async def construct_model(self, params : mHyperParams, in_device : torch.cuda.device): pass

    @abc.abstractmethod
    def set_optimizer(self, opti : mOptimizer = None): pass

    @abc.abstractmethod
    def set_dataset(self, dataset : mDataSet = None): pass

    @abc.abstractmethod
    def load_model(self, in_model : mNeuralNet): pass

    @abc.abstractmethod
    def train(self): pass

    @abc.abstractmethod
    def calc(self): pass

class my_oscil_net(abs_neural_net):

    mymodel = None
    mydevice = None
    myoptimizer = None

    x0_true = None
    dx0dt_true = None


    class mySpecialDataSet(mDataSet_mongo):

        def pde(self, out, t, nu=3):
            omega = 2 * torch.pi * nu
            dxdt = torch.autograd.grad(out, t, torch.ones_like(t), create_graph=True, \
                                    retain_graph=True)[0]
            d2xdt2 = torch.autograd.grad(dxdt, t, torch.ones_like(t), create_graph=True, \
                                    retain_graph=True)[0]
            f = d2xdt2 + (omega ** 2) * out
            return f

        def loss_function(self, t, obj):
            outer_instance = obj
            metric_data = nn.MSELoss()

            out = outer_instance.mymodel(t).to(outer_instance.mydevice)
            f1 = self.pde(out, t, self.params['nu'])

            inlet_mask = (t[:, 0] == 0)
            t0 = t[inlet_mask]
            x0 = outer_instance.mymodel(t0).to(outer_instance.mydevice)
            dx0dt = torch.autograd.grad(x0, t0, torch.ones_like(t0), create_graph=True, \
                                retain_graph=True)[0]

            loss_bc = metric_data(x0, outer_instance.x0_true) + \
                        metric_data(dx0dt, outer_instance.dx0dt_true.to(outer_instance.mydevice))
            loss_pde = metric_data(f1, torch.zeros_like(f1))

            loss = 1e3*loss_bc + loss_pde

            return loss

    def load_model(self, in_model : mNeuralNet): pass



    def set_dataset(self, dataset : mDataSet = None):
        if dataset is None:
            self.neural_model.data_set = [self.mySpecialDataSet(
                                                                power_time_vector=self.neural_model.hyper_param.power_time_vector,
                                                                params={'nu': 3}
                                                                )
                                          ]

    def set_optimizer(self, opti : mOptimizer = None):
        if opti is None:
            self.neural_model.optimizer = [mOptimizer_mongo(method='LBFGS', params={'lr':0.1})]
            self.myoptimizer = torch.optim.LBFGS(self.mymodel.parameters(), lr=0.1)


    async def construct_model(self, params : mHyperParams, in_device : torch.cuda.device):
        await self.createModel(params)

        self.mydevice = in_device

        self.x0_true = torch.tensor([1], dtype=float).float().to(self.mydevice)
        self.dx0dt_true = torch.tensor([0], dtype=float).float().to(self.mydevice)

        self.mymodel = simpleModel(hidden_size=self.neural_model.hyper_param.hidden_size).to(self.mydevice)
        self.set_optimizer()

    def item_closure(self, t):
        self.myoptimizer.zero_grad()
        loss = self.neural_model.data_set[0].loss_function(t, self)
        loss.backward()
        return loss

    def train(self):
        steps = self.neural_model.hyper_param.epochs
        power_of_input = self.neural_model.data_set[0].power_time_vector

        print('power_of_input train', type(power_of_input), power_of_input)
        pbar = tqdm(range(steps), desc='Training Progress')
        t = (torch.linspace(0, 1, power_of_input).unsqueeze(1)).to(self.mydevice)
        t.requires_grad = True


        closure = lambda : self.item_closure(t)

        for step in pbar:


            self.myoptimizer.step(closure)
            if step % 2 == 0:
                current_loss = closure().item()
                pbar.set_description("Step: %d | Loss: %.6f" %
                                     (step, current_loss))



    def calc(self):
        print('calc', self.neural_model.model_dump() )

        power_of_input = self.neural_model.data_set[0].power_time_vector
        nu = self.neural_model.data_set[0].params['nu']
        print('nu', nu)
        print('power_of_input calc', type(power_of_input), power_of_input)
        t = torch.linspace(0, 1, power_of_input).unsqueeze(-1).unsqueeze(0).to(self.mydevice)
        t.requires_grad = True

        x_pred = self.mymodel(t.float())

        omega = 2 * torch.pi * nu
        x_true = self.x0_true * torch.cos(omega*t)
        # print('x_pred', x_pred[0].cpu().detach().numpy())
        # print('x_true', x_true[0].cpu().detach().numpy())

        fs = 13
        plt.scatter(t[0].cpu().detach().numpy(), x_pred[0].cpu().detach().numpy(), label='pred',
                    marker='o',
                    alpha=.7,
                    s=50)
        plt.plot(t[0].cpu().detach().numpy(), x_true[0].cpu().detach().numpy(),
                 color='blue',
                 label='analytical')
        plt.xlabel('t', fontsize=fs)
        plt.ylabel('x(t)', fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.legend()
        plt.title('x(t)')
        # plt.savefig('x.png')
        # plt.show()

        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()


        plt.clf()

        return my_base64_jpgData





class neural_net_microservice():
    inner_model : abs_neural_net = None
    models_list = {'oscil': my_oscil_net}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def create_model(self, params : mHyperParams):
        if (params.mymodel_type is not None):
            if (params.mymodel_type in self.models_list):
                self.inner_model = (self.models_list[params.mymodel_type])()
                await self.inner_model.construct_model(params, self.device)

                print('inner model', self.inner_model)

    async def train_model(self):
        if (self.inner_model is not None):
            self.inner_model.train()


    def run_model(self):
        base64_encoded_image = b''
        if (self.inner_model is not None):
            base64_encoded_image = self.inner_model.calc()

        return base64_encoded_image