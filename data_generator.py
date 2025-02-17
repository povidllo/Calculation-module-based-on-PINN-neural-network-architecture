import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from mongo_schemas import *


def oscillator(x, d=2, w0=20):
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

def generator(num_t, num_ph, typ='train'):
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

def data_generator(dataset : mDataSet):
    x, x_data, x_physics = generator(dataset[0].num_dots[0], dataset[0].num_dots[1])
    x = torch.FloatTensor(x)
    x_data = torch.FloatTensor(x_data)
    x_physics = torch.FloatTensor(x_physics).requires_grad_(True)
    y = oscillator(x).view(-1,1)
    y_data = y[0:len(x)//2:len(x)//20]

    np.save(sys.path[0] + '/data/OSC.npy', y.cpu().detach().numpy())
    
    plt.figure()
    plt.plot(x, y, label="Exact solution")
    plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()
    
    return {'main': x_physics, 'secondary': x_data, 'secondary_true': y_data}

    # self.x, self.x_data, self.x_physics = dg.generator(self.num_t, self.num_ph)
    # self.x = torch.FloatTensor(self.x)
    # self.x_data = torch.FloatTensor(self.x_data)
    # self.x_physics = torch.FloatTensor(self.x_physics).requires_grad_(True)
    # self.y = oscillator(self.x).view(-1,1)
    # self.y_data = self.y[0:len(self.x)//2:len(self.x)//20]
    # plt.figure()
    # plt.plot(self.x, self.y, label="Exact solution")
    # plt.scatter(self.x_data, self.y_data, color="tab:orange", label="Training data")
    # plt.legend()
    # plt.show()