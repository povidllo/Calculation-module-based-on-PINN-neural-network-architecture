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

class oscillator_nn(abs_neural_net):
    mymodel = None
    mydevice = None
    myoptimizer = None
    
    loss_history = []
    l2_history = []
    best_loss = float('inf')
    best_epoch = 0    
    
    class mySpecialDataSet(mDataSet_mongo):
        def oscillator(self, x, d=2, w0=20):
            assert d < w0
            w = np.sqrt(w0**2-d**2)
            phi = np.arctan(-d/w)
            A = 1/(2*np.cos(phi))
            cos = torch.cos(phi+w*x)
            sin = torch.sin(phi+w*x)
            exp = torch.exp(-d*x)
            y = exp*2*A*cos
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

        def data_generator(self):
            x, x_data, x_physics = self.generator(self.num_dots[0], self.num_dots[1])
            x = torch.FloatTensor(x)
            x_data = torch.FloatTensor(x_data)
            x_physics = torch.FloatTensor(x_physics).requires_grad_(True)
            y = self.oscillator(x).view(-1,1)
            y_data = y[0:len(x)//2:len(x)//20]

            # Сохраняем данные в base64
            buffer = io.BytesIO()
            np.savez_compressed(buffer, 
                x=x.cpu().detach().numpy(),
                x_data=x_data.cpu().detach().numpy(),
                x_physics=x_physics.cpu().detach().numpy(),
                y=y.cpu().detach().numpy(),
                y_data=y_data.cpu().detach().numpy()
            )
            encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Сохраняем в базу данных
            self.params['points_data'] = encoded_data
            
            # Для обратной совместимости пока оставляем и старое сохранение
            np.save(sys.path[0] + '/data/OSC.npy', y.cpu().detach().numpy())
            
            return {'main': x_physics, 'secondary': x_data, 'secondary_true': y_data}

        def load_data_from_params(self):
            """Загрузка данных из base64 в params"""
            if 'points_data' not in self.params:
                return None
                
            # Декодируем данные из base64
            decoded_data = base64.b64decode(self.params['points_data'])
            buffer = io.BytesIO(decoded_data)
            data = np.load(buffer)
            
            # Преобразуем обратно в тензоры
            x = torch.FloatTensor(data['x'])
            x_data = torch.FloatTensor(data['x_data'])
            x_physics = torch.FloatTensor(data['x_physics']).requires_grad_(True)
            y = torch.FloatTensor(data['y'])
            y_data = torch.FloatTensor(data['y_data'])
            
            return {'main': x_physics, 'secondary': x_data, 'secondary_true': y_data}

        def equation(self,yhp, x_physics, d=2, w0=20):
            mu = d * 2
            k = w0 ** 2
            dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
            dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]
            physics = dx2 + mu*dx + k*yhp
            return physics


        def loss_calculator(self,yhp, x_physics, yh, y_data):
            # loss_f = torch.mean(ac_equation(u_pred_f, variables_f) ** 2)
            # loss_u = torch.mean((u_pred - u_data) ** 2)
            # loss = loss_f + loss_u
            loss1 = torch.mean((yh-y_data)**2)# use mean squared error
            
            physics = self.equation(yhp, x_physics)
            loss2 = (1e-4)*torch.mean(physics**2)

            loss = loss1 + loss2    

            return loss     
        
        def calculate_l2_error(self, path_true_data, model, device, test_data_generator):
            x, _, _ = test_data_generator()
            
            # Загружаем данные из базы вместо файла
            if 'points_data' not in self.params:
                print("Warning: No data found in database, using file")
                y = np.load(sys.path[0] + path_true_data)
            else:
                # Декодируем данные из base64
                decoded_data = base64.b64decode(self.params['points_data'])
                buffer = io.BytesIO(decoded_data)
                data = np.load(buffer)
                y = data['y']
            
            u_pred = model(x)
            u_pred = u_pred.cpu().detach().numpy()
            true = y
            # Сравнение с эталоном
            error = np.linalg.norm(u_pred - true, 2) / np.linalg.norm(true, 2)
            return error
                
                
        
    async def load_model(self, in_model : mNeuralNet, in_device):
        load_nn = await mNeuralNet_mongo.get(in_model.stored_item_id, fetch_links=True)
        
        # Всегда создаем новую модель
        self.neural_model = load_nn
        self.neural_model.records = []
        await self.set_dataset()
        
        self.mydevice = in_device
        self.mymodel = pinn(self.neural_model.hyper_param).to(self.mydevice)
        self.set_optimizer()
        
        # Пытаемся загрузить веса, если они есть
        # try:
        #     self.mymodel.load_state_dict(torch.load(sys.path[0] + self.neural_model.hyper_param.save_weights_path))
        #     print("Weights loaded successfully" + self.neural_model.hyper_param.save_weights_path)
        # except:
        #     print("No saved weights found, using initialized weights")
        cur_state = await self.abs_load_weights()
        if (cur_state is not None):
            self.mymodel.load_state_dict(cur_state)
        else:
            print("No saved weights found, using initialized weights")




    async def set_dataset(self, dataset : mDataSet = None):
        if dataset is None:
            self.neural_model.data_set = [self.mySpecialDataSet(
                                                                power_time_vector=self.neural_model.hyper_param.power_time_vector,
                                                                params={'nu': 3},
                                                                num_dots=self.neural_model.hyper_param.num_dots
                                                                )
                                          ]
        else:
            new_dataset = self.mySpecialDataSet(
                power_time_vector=self.neural_model.hyper_param.power_time_vector,
                params=dataset.params
            )
            await self.update_dataset_for_nn(new_dataset)
        
        # Пробуем загрузить данные из базы
        loaded_data = self.neural_model.data_set[0].load_data_from_params()
        
        if loaded_data is None:
            # Если данных в базе нет, генерируем новые
            data = self.neural_model.data_set[0].data_generator()
            print("loaded_data is none")
        else:
            data = loaded_data
            print("loaded_data is not none")
            
        self.variables_f = data['main'].to(self.mydevice)
        self.u_data = data['secondary_true'].to(self.mydevice)
        self.variables = data['secondary'].to(self.mydevice)

    
    def set_optimizer(self, opti : mOptimizer = None):
        if opti is None:
            self.neural_model.optimizer = [mOptimizer_mongo(method='Adam', params={'lr':0.1})]
            self.myoptimizer = create_optim(self.mymodel, get_config())
            
    async def construct_model(self, params : mHyperParams, in_device ):
        await self.createModel(params)

        self.mydevice = in_device
        self.mymodel = pinn(params).to(self.mydevice)
        self.set_optimizer()
    

    # def save_model(self, path):
    #     """Сохранение модели"""
    #     torch.save(self.mymodel.state_dict(), path)
    async def save_model(self, path):
        print('run saving')
        weights = self.mymodel.state_dict()
        await self.abs_save_weights(weights)
        print('save complited')
    
    async def train(self):
        cfg = self.neural_model.hyper_param
        epochs = cfg.epochs
        self.config = self.neural_model.hyper_param

        for epoch in tqdm(range(epochs)):
            self.myoptimizer.zero_grad()
            u_pred = self.mymodel(self.variables)
            u_pred_f = self.mymodel(self.variables_f)
            
            loss = self.neural_model.data_set[0].loss_calculator(u_pred_f, self.variables_f, u_pred, self.u_data)
            loss.backward()
            self.myoptimizer.step()

            current_loss = loss.item()
            self.loss_history.append(current_loss)
            
            if self.neural_model.data_set[0].calculate_l2_error:
                l2_error = self.neural_model.data_set[0].calculate_l2_error(
                            self.config.path_true_data, 
                            self.mymodel, 
                            self.mydevice, 
                            test_data_generator)
                self.l2_history.append(l2_error)            

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                
                # self.save_model(sys.path[0] + self.config.save_weights_path)
            
            if(epoch % 400 == 0):
                print(f"Epoch {epoch}, Train loss: {current_loss}, L2: {l2_error if self.neural_model.data_set[0].calculate_l2_error else 0}")
                
        await self.save_model(sys.path[0] + self.config.save_weights_path)




    async def calc(self):
        self.mymodel.load_state_dict(torch.load(sys.path[0] + self.neural_model.hyper_param.save_weights_path))
        
        x, _, _ = test_data_generator()
        
        # Загружаем данные из базы вместо файла
        if 'points_data' not in self.neural_model.data_set[0].params:
            print("Warning: No data found in database, using file")
            y = np.load(sys.path[0] + self.neural_model.hyper_param.path_true_data)
        else:
            decoded_data = base64.b64decode(self.neural_model.data_set[0].params['points_data'])
            buffer = io.BytesIO(decoded_data)
            data = np.load(buffer)
            y = data['y']
            
        u_pred = self.mymodel(x)
        u_pred = u_pred.cpu().detach().numpy()
        true = y

        plt.figure(figsize=(10, 6))
        
        # Преобразуем x в numpy array если это тензор
        x_plot = x.cpu().numpy() if torch.is_tensor(x) else x
        
        # Строим оба графика
        plt.plot(x_plot, true, 'b-', linewidth=2, label='Истинное решение')
        plt.plot(x_plot, u_pred, 'r--', linewidth=2, label='Предсказание модели')
        
        # Настройки графика
        plt.xlabel('Временная координата (t)', fontsize=12)
        plt.ylabel('Значение y', fontsize=12)
        plt.title('Сравнение предсказаний модели с эталонным решением', fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Рассчитываем и выводим ошибку
        error = np.linalg.norm(u_pred - true, 2) / np.linalg.norm(true, 2)
        plt.text(0.05, 0.95, f'Средняя абсолютная ошибка: {error:.4f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

        new_rec = mongo_Record(record={'raw' : my_base64_jpgData})
        await self.append_rec_to_nn(new_rec)


        plt.clf()

        return my_base64_jpgData
        # plt.show()
