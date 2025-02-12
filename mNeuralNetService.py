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

# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
#
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
#
#
# from torchviz import make_dot
#
# from torchvision import models
# from torchsummary import summary
# import hiddenlayer as hl


class abs_neural_net(abc.ABC):
    neural_model : mNeuralNet_mongo = None


    async def createModel(self,params : mHyperParams):
        self.neural_model = mNeuralNet_mongo(hyper_param=mHyperParams_mongo(**params.model_dump()))
        await mNeuralNet_mongo.m_insert(self.neural_model)

    @abc.abstractmethod
    async def construct_model(self, params : mHyperParams): pass

    @abc.abstractmethod
    def set_optimizer(self, opti : mOptimizer): pass

    @abc.abstractmethod
    def set_dataset(self, dataset : mDataSet): pass

    @abc.abstractmethod
    def load_model(self, model : mNeuralNet): pass

    @abc.abstractmethod
    def train(self): pass

    @abc.abstractmethod
    def calc(self): pass

class my_oscil_net(abs_neural_net):

    def __init__(self):
        pass

    def load_model(self, model : mNeuralNet): pass

    def set_dataset(self, dataset : mDataSet): pass

    def set_optimizer(self, opti : mOptimizer): pass

    async def construct_model(self, params : mHyperParams):
        await self.createModel(params)




    def train(self): pass

    def calc(self):
        return self.neural_model.model_dump()


class neural_net_microservice():
    inner_model : abs_neural_net = None
    models_list = {'oscil': my_oscil_net}

    async def create_model(self, params : mHyperParams):
        if (params.mymodel_type is not None):
            if (params.mymodel_type in self.models_list):
                self.inner_model = (self.models_list[params.mymodel_type])()
                await self.inner_model.construct_model(params)

                print('inner model', self.inner_model)

    def run_model(self):
        ret = ''
        if (self.inner_model is not None):
            ret = str(self.inner_model.calc())
            print('run_model', ret)
        return ret