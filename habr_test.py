from contextlib import asynccontextmanager
from fastapi import Body
import json
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi import FastAPI, BackgroundTasks, Request, Depends,WebSocket, WebSocketDisconnect, Query, Response, APIRouter
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
import fastapi_jsonrpc as jsonrpc

import jinja2
import io
import base64

import mTemplate
from mongo_schemas import *
from mNeuralNetService import neural_net_microservice


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

async def create_neural_model(model_type : Optional[str] = None, desc : Optional[str] = 'hello world'):
    if (model_type is not None):
        params = mHyperParams(mymodel_type=model_type, mymodel_desc=desc)
        await neural_net_manager.create_model(params)

    return {"resp" : "OK"}

async def run_neural_net():
    res = neural_net_manager.run_model()

    return {"result": res}

router = APIRouter()

def main(loop):

    async def init():
        await init_beanie(database=client[database_name], document_models=[mongo_Estimate, mongo_Record, mOptimizer_mongo, mDataSet_mongo, mHyperParams_mongo, mNeuralNet_mongo])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init()
        yield

    app = FastAPI(lifespan=lifespan)

    app.add_api_route("/create_model", create_neural_model, methods=["GET"])
    app.add_api_route("/run", run_neural_net, methods=["GET"])


    app.include_router(router)

    import uvicorn

    config = uvicorn.Config(app, host=cur_host, port=8010, loop=loop,access_log=False)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())

if __name__ == '__main__':
    client = AsyncIOMotorClient("mongodb://"+database_ulr+":27017")
    neural_net_manager = neural_net_microservice()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main(loop)


# import torch
# import torch.nn as nn
#
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
#
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# # from google.colab import files as filescolab
#
#
# from torchviz import make_dot
#
# from torchvision import models
# from torchsummary import summary
# import hiddenlayer as hl


# client = AsyncIOMotorClient("mongodb://"+database_ulr+":27017")
#
# async def init():
#     await init_beanie(database=client[database_name], document_models=[mongo_Estimate, mongo_Record])
#
# torch.manual_seed(20)
#
# api_v1 = jsonrpc.Entrypoint('/api/v1/jsonrpc')
#
# class simpleModel(nn.Module):
#   def __init__(self,
#                hidden_size=20):
#     super().__init__()
#     self.layers_stack = nn.Sequential(
#         nn.Linear(1, hidden_size),
#         nn.Tanh(),
#         nn.Linear(hidden_size, hidden_size), #1
#         nn.Tanh(),
#         nn.Linear(hidden_size, hidden_size), #2
#         nn.Tanh(),
#         nn.Linear(hidden_size, hidden_size), #3
#         nn.Tanh(),
#         nn.Linear(hidden_size, hidden_size), #4
#         nn.Tanh(),
#         nn.Linear(hidden_size, 1),
#         nn.Tanh(),
#     )
#
#
#   def forward(self, x):
#     return self.layers_stack(x)
#
# isTrained = False
# isTraining = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# x0_true=torch.tensor([1], dtype=float).float().to(device)
# dx0dt_true = torch.tensor([0], dtype=float).float().to(device)
# model = simpleModel().to(device)
#
# def pde(out, t, nu=3):
#     omega = 2 * torch.pi * nu
#     dxdt = torch.autograd.grad(out, t, torch.ones_like(t), create_graph=True, \
#                             retain_graph=True)[0]
#     d2xdt2 = torch.autograd.grad(dxdt, t, torch.ones_like(t), create_graph=True, \
#                             retain_graph=True)[0]
#     f = d2xdt2 + (omega ** 2) * out
#     return f
#
#
# def pdeBC(t):
#     metric_data = nn.MSELoss()
#
#     out = model(t).to(device)
#     f1 = pde(out, t)
#
#     inlet_mask = (t[:, 0] == 0)
#     t0 = t[inlet_mask]
#     x0 = model(t0).to(device)
#     dx0dt = torch.autograd.grad(x0, t0, torch.ones_like(t0), create_graph=True, \
#                         retain_graph=True)[0]
#
#     loss_bc = metric_data(x0, x0_true) + \
#                 metric_data(dx0dt, dx0dt_true.to(device))
#     loss_pde = metric_data(f1, torch.zeros_like(f1))
#
#     loss = 1e3*loss_bc + loss_pde
#
#     return loss
#
#
# def train(in_steps):
#     global isTrained
#     global isTraining
#     isTraining = True
#     steps = in_steps
#     pbar = tqdm(range(steps), desc='Training Progress')
#     optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
#     t = (torch.linspace(0, 1, 100).unsqueeze(1)).to(device)
#     t.requires_grad = True
#
#     for step in pbar:
#         def closure():
#             optimizer.zero_grad()
#             loss = pdeBC(t)
#             loss.backward()
#             return loss
#
#         optimizer.step(closure)
#         if step % 2 == 0:
#             current_loss = closure().item()
#             pbar.set_description("Step: %d | Loss: %.6f" %
#                                  (step, current_loss))
#     isTrained = True
#     isTraining = False
#
#
#
#
# async def predict():
#     nu = 3
#     omega = 2 * torch.pi * nu
#
#
#     t = torch.linspace(0, 1, 100).unsqueeze(-1).unsqueeze(0).to(device)
#     t.requires_grad = True
#     x_pred = model(t.float())
#     # print('x_pred ', type(x_pred.tolist()), x_pred.tolist())
#
#
#
#     # est = mongo_Estimate(records=[mongo_Record(record=x_pred.tolist())])
#     # await mongo_Estimate.m_insert(est)
#     # print('est ', type(est), est)
#
#     x_true = x0_true * torch.cos(omega*t)
#
#     fs = 13
#     plt.scatter(t[0].cpu().detach().numpy(), x_pred[0].cpu().detach().numpy(), label='pred',
#                 marker='o',
#                 alpha=.7,
#                 s=50)
#     plt.plot(t[0].cpu().detach().numpy(), x_true[0].cpu().detach().numpy(),
#              color='blue',
#              label='analytical')
#     plt.xlabel('t', fontsize=fs)
#     plt.ylabel('x(t)', fontsize=fs)
#     plt.xticks(fontsize=fs)
#     plt.yticks(fontsize=fs)
#     plt.legend()
#     plt.title('x(t)')
#     # plt.savefig('x.png')
#     # plt.show()
#
#     my_stringIObytes = io.BytesIO()
#     plt.savefig(my_stringIObytes, format='jpg')
#     my_stringIObytes.seek(0)
#     my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
#
#
#     plt.clf()
#
#     # transforms = [hl.transforms.Prune('Constant')]
#     # graph = hl.build_graph(model, batch.text, transforms=transforms)
#     # graph.theme = hl.graph.THEMES['blue'].copy()
#     # graph.save('rnn_hiddenlayer', format='png')
#
#     # vizNN = make_dot(x_pred.mean(), params=dict(model.named_parameters()))
#     # print('vizNN', vizNN)
#     # print('model', model)
#     # my_base64_jpgData = np.concatenate((my_base64_jpgData, vizNN), axis=1)
#
#
#
#
#     return my_base64_jpgData
#
#
#
# async def pinn() :
#     global isTrained
#
#     base64_encoded_image = b''
#
#     loader = jinja2.FileSystemLoader("./templates")
#     env = jinja2.Environment(loader=loader,autoescape = False)
#     table_templ = env.get_template("index.html")
#
#     if isTrained:
#         base64_encoded_image = await predict()
#
#     table_templ = table_templ.render(myImage=base64_encoded_image)
#
#     return HTMLResponse(mTemplate.mtemplate('<script></script>', table_templ))
#
#
# async def train_pinn(background_tasks: BackgroundTasks, N:Optional[int] = None):
#     global isTrained
#     global isTraining
#     steps = 10
#     if (N is not None) and (not isTraining):
#         isTrained = False
#         steps = N
#
#
#     # all_est = await mongo_Estimate.find_entries_all()
#     # print('all_est', all_est[0].id)
#     # ret_all_est = [str(el.id) for el in all_est]
#     # print('ret_all_est', ret_all_est)
#     if (not isTrained) and (not isTraining):
#         background_tasks.add_task(train, steps)
#     # return {"ok" : isTrained , 'training' : isTraining, 'ests': ret_all_est}
#     return {"ok" : isTrained , 'training' : isTraining}
# if __name__ == '__main__':
#
#     @asynccontextmanager
#     async def lifespan(app: jsonrpc.API):
#         await init()
#         yield
#
#
#     app = jsonrpc.API(lifespan=lifespan)
#     app.bind_entrypoint(api_v1)
#     app.add_api_route("/pinn", pinn, methods=["GET"])
#     app.add_api_route("/train", train_pinn, methods=["GET"])
#
#     import uvicorn
#     uvicorn.run(app, host=cur_host, port=8010)