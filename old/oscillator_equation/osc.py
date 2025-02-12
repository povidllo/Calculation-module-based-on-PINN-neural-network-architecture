import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
# !pip install wandb -qU
# !pip install ml_collections
import modulus
import data_generator as dg
import default
import wandb
torch.manual_seed(123)
'''

Уравнение гармонического осцилятора
m*d^2x/d^2t + mu * dx/dt + kx = 0

m = 1  - масса
d(delta) = 2
w0 = 20
mu = 2 * d
k = w0**2
Граничные условия
x0_bc = 1
dx_dt0_bc = 0

'''

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


def equation(yhp, x_physics, d=2, w0=20):
    mu = d * 2
    k = w0 ** 2
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
    dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]
    physics = dx2 + mu*dx + k*yhp
    return physics

class TrainClass:
    def __init__(self, cfg, model, wandbFlag=False):
        self.num_t = cfg.num_t
        self.num_ph = cfg.num_ph
        
        self.num_epochs = cfg.epochs
        self.num_hidden_layers = 4
        self.save_path = cfg.save_path
        
        self.num_nodes = cfg.hidden_sizes
        self.learning_rate = cfg.lr
        self.wandbFlag = wandbFlag
        if self.wandbFlag:
          self.__wandbConnect(cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Operation mode: ", self.device)

        # Данные
        self.__createData()
        
        # Модель
        self.model = model.to(self.device)
        
        # Оптимизатор
        # self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.999, 0.999), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(model.parameters(),betas=(0.9, 0.999),lr=self.learning_rate)
        # Логирование
        self.loss_history = []
        self.l2_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    
    '''    
    Подключается к wandb для отслеживания процесса обучения
    '''
    def __wandbConnect(self, cfg):
        wandb.login()
        wandb.init(
            project=cfg.project,
            name=cfg.name,
            config={
            "epochs": cfg.epochs,
            })
        
    '''
    Генерирует данные:
    variables - выборка, содержищие граничные и начальные условия
    variables_f - выборка из t
    t - содержит все точки оси t
    '''
    def __createData(self):
        self.x, self.x_data, self.x_physics = dg.generator(self.num_t, self.num_ph)
        self.x = torch.FloatTensor(self.x)
        self.x_data = torch.FloatTensor(self.x_data)
        self.x_physics = torch.FloatTensor(self.x_physics).requires_grad_(True)
        self.y = oscillator(self.x).view(-1,1)
        self.y_data = self.y[0:len(self.x)//2:len(self.x)//20]
        plt.figure()
        plt.plot(self.x, self.y, label="Exact solution")
        plt.scatter(self.x_data, self.y_data, color="tab:orange", label="Training data")
        plt.legend()
        plt.show()

    def __calculate_l2_error(self):

        u_pred = self.model(self.x)
        u_pred = u_pred.cpu().detach().numpy()
        true=self.y.cpu().detach().numpy()
        # Сравнение с эталоном
        error = np.linalg.norm(u_pred - true, 2) / np.linalg.norm(true, 2)
        return error

        
    '''
    Выводит график вычисленного уравнения
    xp - тестовые точки для нейросети(для отображения)
    yh - выходные точки нейросети от self.x
    '''
    def printEval(self,epoch=None):
        
        self.model.load_state_dict(torch.load(self.save_path, map_location=torch.device(self.device)))

        print("L2 error: ", self.__calculate_l2_error())
        u_pred = self.model(self.x).detach()
        u_pred = u_pred.cpu().numpy()
        
        xp = self.x_physics.detach()
        
        "Pretty plot training results"
        plt.figure(figsize=(8,4))
        plt.plot(self.x,self.y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(self.x, u_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(self.x_data, self.y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-1.1, 1.1)
        if epoch != None:
            plt.text(1.065,0.7,"Training step: %i"%(epoch+1),fontsize="xx-large",color="k")
        plt.axis("off")

    '''
    Выводит график функции потерь, а также эпоху с наименьшей величиной потерь
    '''
    def printLossGraph(self):
        print(f"[Best][Epoch: {self.best_epoch}] Train loss: {self.best_loss}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title("loss")
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.l2_history)
        plt.title("L2 loss")

        plt.show()
        
    def train(self):
        d, w0 = 2, 20
        mu, k = 2*d, w0**2

        for epoch in tqdm(range(self.num_epochs)):
            self.optimizer.zero_grad()
            
            # compute the "data loss"
            yh = self.model(self.x_data)
            loss1 = torch.mean((yh-self.y_data)**2)# use mean squared error
            
            # compute the "physics loss"
            yhp = self.model(self.x_physics)
            physics = equation(yhp, self.x_physics)
            loss2 = (1e-4)*torch.mean(physics**2)
            
        
            loss = loss1 + loss2
            if(epoch % 400 == 0):
                print(epoch, loss.item(), self.__calculate_l2_error())
            
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            self.loss_history.append(current_loss)
            l2_error = self.__calculate_l2_error()
            self.l2_history.append(l2_error)

            # Сохранение лучшей модели
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.save_path)

            
            if epoch:
                # print(f"Epoch {epoch}, Train loss: {current_loss}, L2: {l2_error}")
                
                if self.wandbFlag:
                  wandb.log({"epoche": epoch, "loss": current_loss})
                  wandb.log({"epoche": epoch, "L2": l2_error})
            if epoch % 1000 == 0:   
                self.printEval(epoch)
                
        self.printLossGraph()

import default
import cfg_test1
import cfg_test2
import cfg_test3
import cfg_test4

hyperparameters = default.get_config()
# hyperparameters = cfg_test1.get_config()
# hyperparameters = cfg_test2.get_config()
# hyperparameters = cfg_test3.get_config()
# hyperparameters = cfg_test4.get_config()

model = modulus.pinn(hyperparameters)
a = TrainClass(hyperparameters, model)
a.train()
# a.printEval()
# a.printLossGraph()
# a.train()