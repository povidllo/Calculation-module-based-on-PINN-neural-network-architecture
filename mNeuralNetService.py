from fastapi import WebSocket
import asyncio
import base64
import torch

from mNeural_abs import *
from oscillator import oscillator_nn


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, usr_name: str):
        await websocket.accept()
        self.active_connections.setdefault(usr_name, []).append(websocket)

    def disconnect(self, websocket: WebSocket, usr_name: str):
        if usr_name in self.active_connections:
            self.active_connections[usr_name].remove(websocket)
            if not self.active_connections[usr_name]:
                del self.active_connections[usr_name]

    async def send_personal_message_json(self, message: ChatMessage):
        for conn in self.active_connections.get(message.user, []):
            await conn.send_json(message.model_dump(), mode='text')


class NeuralNetMicroservice:
    def __init__(self):
        self.inner_model: AbsNeuralNet | None = None
        self.models_list = {'oscil': oscillator_nn}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ws_manager = ConnectionManager()

    async def create_model(self, params: mHyperParams):
        print(f"params: {params}")
        model_type = params.my_model_type
        if model_type and model_type in self.models_list:
            self.inner_model = self.models_list[model_type]()
            await self.inner_model.construct_model(params, self.device)

    async def train_model(self):
        if self.inner_model:
            print(self.inner_model)
            await self.inner_model.train()

    async def set_dataset(self, dataset: mDataSet | None = None):
        if dataset and self.inner_model:
            await self.inner_model.set_dataset(dataset)

    async def load_nn(self, inp_nn: mNeuralNetMongo | None = None):
        my_nn = await mNeuralNetMongo.get_item_by_id(inp_nn)
        print(f"my_nn - {type(my_nn)} {my_nn.model_dump()}")

        params = await my_nn.hyper_param.fetch()
        print(f"hyper_param - {type(params)} {params}")

        model_type = params.my_model_type
        if model_type and model_type in self.models_list:
            self.inner_model = self.models_list[model_type]()
            await self.inner_model.load_model(inp_nn, self.device)

        return params.__dict__

    async def run_model(self):
        if self.inner_model:
            return await self.inner_model.calc()
        return b''
