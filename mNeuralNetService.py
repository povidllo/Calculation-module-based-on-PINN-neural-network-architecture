from fastapi import WebSocket
import torch

from mNeural_abs import ChatMessage, AbsNeuralNet, mHyperParams, mDataSet, mNeuralNetMongo, Optional
from equations.oscillator_eq.oscillator import oscillator_nn
from equations.allen_cahn_eq.allen_cahn import allen_cahn_nn


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, usr_name: str):
        await websocket.accept()
        if usr_name in self.active_connections:
            self.active_connections[usr_name].append(websocket)
        else:
            self.active_connections[usr_name] = [websocket]

    def disconnect(self, websocket: WebSocket, usr_name: str):
        if usr_name in self.active_connections:
            self.active_connections[usr_name].remove(websocket)

    async def send_personal_message_json(self, message: ChatMessage):
        if message.user in self.active_connections:
            for el in self.active_connections[message.user]:
                await el.send_json(message.model_dump(), mode='text')


class NeuralNetMicroservice:
    inner_model: AbsNeuralNet = None
    models_list = {'oscil': oscillator_nn, 'allencahn': allen_cahn_nn}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_manager = ConnectionManager()

    async def create_model(self, params: mHyperParams):
        print(f"PARAMS: {params}")
        if params.my_model_type is not None and params.my_model_type in self.models_list:
            self.inner_model = (self.models_list[params.my_model_type])()
            await self.inner_model.construct_model(params, self.device)

    async def train_model(self):
        if self.inner_model is not None:
            print(self.inner_model)
            await self.inner_model.train()

    async def set_dataset(self, dataset: mDataSet = None):
        if dataset is not None:
            await self.inner_model.set_dataset(dataset)

    async def load_nn(self, inp_nn: Optional[mNeuralNetMongo] = None):

        my_nn = await mNeuralNetMongo.get_item_by_id(inp_nn)

        print(f"\n\nТуточки \n{my_nn.model_dump()} \nценок\n")

        print(f"my_nn - {type(my_nn)} {my_nn.model_dump()}")
        print(f"hyper_param1 - {type(my_nn.hyper_param)}, {my_nn.hyper_param}")
        params = await my_nn.hyper_param.fetch()
        print(f"hyper_param2 - {type(params)} {params}")

        if params.my_model_type is not None and params.my_model_type in self.models_list:
            self.inner_model = (self.models_list[params.my_model_type])()
            await self.inner_model.load_model(inp_nn, self.device)

        return params.__dict__

    async def run_model(self):
        base64_encoded_image = b''
        if self.inner_model is not None:
            base64_encoded_image = await self.inner_model.calc()


        return base64_encoded_image
