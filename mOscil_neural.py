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

import sys


class oscillator(abs_neural_net):
    mymodel = None
    mydevice = None
    myoptimizer = None
    
    class mySpecialDataSet(mDataSet_mongo):
        def oscillator(self, x, d=2, w0=20):
            assert d < w0
            w = np.sqrt(w0**2-d**2)
            phi = np.arctan(-d/w)
            A = 1/(2*np.cos(phi))
            cos = torch.cos(phi+w*x)
            sin = torch.sin(phi+w*x)
            exp = torch.exp(-d*x)
            y  = exp*2*A*cos
            return y

        def generator(self, num_t, num_ph, typ='train'):
            '''
            Генерирует данные для осцилятора
            t - вся выборка
            t_phys - выборка для обучения физического аспекта модели
            t_data - выборка для обучения путем сравнения с правильными данными
            '''
            t = np.linspace(0, 1, num_t).reshape(-1, 1)
            l = len(t)//2
            t_data = t[0:l:l//10]
            
            t_phys = np.linspace(0, 1, num_ph).reshape(-1, 1)
            return t, t_data, t_phys

        def data_generator(self, cfg):
            x, x_data, x_physics = self.generator(cfg.num_dots[0], cfg.num_dots[1])
            x = torch.FloatTensor(x)
            x_data = torch.FloatTensor(x_data)
            x_physics = torch.FloatTensor(x_physics).requires_grad_(True)
            y = oscillator(x).view(-1,1)
            y_data = y[0:len(x)//2:len(x)//20]

            np.save(sys.path[0] + './OSC.npy', y.cpu().detach().numpy())
            
            plt.figure()
            plt.plot(x, y, label="Exact solution")
            plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
            plt.legend()
            plt.show()
            
            return {'main': x_physics, 'secondary': x_data, 'secondary_true': y_data}

    
            