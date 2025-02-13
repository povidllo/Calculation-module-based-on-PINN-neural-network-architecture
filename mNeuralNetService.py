import jinja2
import io
import base64

from mongo_schemas import *
from mNeural_abs import *
from mHabr_neural_exmp import *

import asyncio
import abc

import torch


class neural_net_microservice():
    inner_model : abs_neural_net = None
    models_list = {'oscil': my_oscil_net}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def create_model(self, params : mHyperParams):
        if (params.mymodel_type is not None):
            if (params.mymodel_type in self.models_list):
                self.inner_model = (self.models_list[params.mymodel_type])()
                await self.inner_model.construct_model(params, self.device)

    async def train_model(self):
        if (self.inner_model is not None):
            self.inner_model.train()

    async def set_dataset(self, dataset : mDataSet = None):
        if (dataset is not None):
            await self.inner_model.set_dataset(dataset)

    async def run_model(self):
        base64_encoded_image = b''
        if (self.inner_model is not None):
            base64_encoded_image = await self.inner_model.calc()

        return base64_encoded_image