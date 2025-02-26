import jinja2
import io
import base64
from fastapi import FastAPI, BackgroundTasks, Request, Depends,WebSocket, Query, Response, APIRouter
from fastapi.websockets import WebSocketState, WebSocketDisconnect
from mongo_schemas import *
from mNeural_abs import *
from mHabr_neural_exmp import my_oscil_net
from oscillator import oscillator_nn

import asyncio
import abc

import torch

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[ list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, usr_name : str):
        await websocket.accept()
        if (usr_name in self.active_connections):
            self.active_connections[usr_name].append(websocket)
        else:
            self.active_connections[usr_name] = [websocket]

    def disconnect(self, websocket: WebSocket, usr_name : str):

        if (usr_name in self.active_connections):
            self.active_connections[usr_name].remove(websocket)

    async def send_personal_message_json(self, message: chatMessage):
        if (message.user in self.active_connections):
            for el in self.active_connections[message.user]:
                await el.send_json(message.model_dump(), mode='text')





class neural_net_microservice():
    inner_model : abs_neural_net = None
    # inner_model -> {'id_nn' : abs_neural_net}
    # models_list = {'oscil': my_oscil_net}
    models_list = {'oscil': oscillator_nn}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_manager = ConnectionManager()

    async def create_model(self, params : mHyperParams):
        print('params', params)
        if (params.mymodel_type is not None):
            if (params.mymodel_type in self.models_list):
                self.inner_model = (self.models_list[params.mymodel_type])()
                await self.inner_model.construct_model(params, self.device)

    async def train_model(self, train_params : mTrainParams):
        print('train_params', train_params)
        if (self.inner_model is not None):
            # Проверяем существует ли уже параметры обучения в базе данных
            if self.inner_model.neural_model.train_param is None:
                # Создаем новый документ с параметрами обучения
                train_params_mongo = mTrainParams_mongo(
                    epochs=train_params.epochs,
                    optimizer=train_params.optimizer,
                    optimizer_lr=train_params.optimizer_lr
                )
                # Сохраняем параметры в базу данных
                await mTrainParams_mongo.insert(train_params_mongo)
                
                # Обновляем ссылку на параметры в модели
                self.inner_model.neural_model.train_param = train_params_mongo
            else:
                # Обновляем существующие параметры
                self.inner_model.neural_model.train_param.epochs = train_params.epochs
                self.inner_model.neural_model.train_param.optimizer = train_params.optimizer
                self.inner_model.neural_model.train_param.optimizer_lr = train_params.optimizer_lr
                
            # Сохраняем изменения в модели
            await mNeuralNet_mongo.m_save(self.inner_model.neural_model)
            
            # Запускаем тренировку
            self.inner_model.train(train_params)

    async def set_dataset(self, dataset : mDataSet = None):
        if (dataset is not None):
            await self.inner_model.set_dataset(dataset)


    async def load_nn(self, inp_nn : Optional[mNeuralNet] = None):

        my_nn = await mNeuralNet_mongo.get_item_by_id(inp_nn)

        print('my_nn-', type(my_nn), my_nn.model_dump())
        print('hyper_param1 - ', type(my_nn.hyper_param), my_nn.hyper_param)
        params = await my_nn.hyper_param.fetch()
        print('hyper_param2 - ' , type(params), params)


        if (params.mymodel_type is not None):
            if (params.mymodel_type in self.models_list):
                self.inner_model = (self.models_list[params.mymodel_type])()
                await self.inner_model.load_model(inp_nn, self.device)


    async def run_model(self):
        base64_encoded_image = b''
        if (self.inner_model is not None):
            base64_encoded_image = await self.inner_model.calc()

        return base64_encoded_image