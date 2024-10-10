from contextlib import asynccontextmanager
from fastapi import Body
import json
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi import FastAPI, BackgroundTasks, Request, Depends,WebSocket, WebSocketDisconnect, Query, Response
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
import fastapi_jsonrpc as jsonrpc
import jinja2
import io
import base64

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from google.colab import files as filescolab



cur_host = 'localhost'
api_v1 = jsonrpc.Entrypoint('/api/v1/jsonrpc')

class MyError(jsonrpc.BaseError):
    CODE = 5000
    MESSAGE = 'My error'

    class DataModel(BaseModel):
        details: str


mtemplate = lambda gscripts, gdivs: """
<!DOCTYPE html>
<html lang="en">
    <head>
        <script>
            var url_login = "https://""" + str(cur_host) + """:8008/login"

            async function login_auth(){
            return await fetch(url_login, {
                    method: 'POST',
                    mode: 'cors',
                    credentials: 'include',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ "username": cur_key,  "password" : cur_val })
                }).then((response) => response.json())
            }
        </script>    
        """ + str(gscripts) + """
    </head>
    <body>
        """ + str(gdivs) + """
        <div id="myplot"></div><p>
    </body>
</html>
"""


class simpleModel(nn.Module):
  def __init__(self,
               hidden_size=20):
    super().__init__()
    self.layers_stack = nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #1
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #2
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #3
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #4
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
        nn.Tanh(),
    )


  def forward(self, x):
    return self.layers_stack(x)

isTrained = False
isTraining = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x0_true=torch.tensor([1], dtype=float).float().to(device)
dx0dt_true = torch.tensor([0], dtype=float).float().to(device)
model = simpleModel().to(device)

def pde(out, t, nu=2):
    omega = 2 * torch.pi * nu
    dxdt = torch.autograd.grad(out, t, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    d2xdt2 = torch.autograd.grad(dxdt, t, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    f = d2xdt2 + (omega ** 2) * out
    return f


def pdeBC(t):
    metric_data = nn.MSELoss()



    out = model(t).to(device)
    f1 = pde(out, t)

    inlet_mask = (t[:, 0] == 0)
    t0 = t[inlet_mask]
    x0 = model(t0).to(device)
    dx0dt = torch.autograd.grad(x0, t0, torch.ones_like(t0), create_graph=True, \
                        retain_graph=True)[0]

    loss_bc = metric_data(x0, x0_true) + \
                metric_data(dx0dt, dx0dt_true.to(device))
    loss_pde = metric_data(f1, torch.zeros_like(f1))

    loss = 1e3*loss_bc + loss_pde

    return loss


def train():
    global isTrained
    global isTraining
    isTraining = True
    steps = 10
    pbar = tqdm(range(steps), desc='Training Progress')
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
    t = (torch.linspace(0, 1, 100).unsqueeze(1)).to(device)
    t.requires_grad = True

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = pdeBC(t)
            loss.backward()
            return loss

        optimizer.step(closure)
        if step % 2 == 0:
            current_loss = closure().item()
            pbar.set_description("Step: %d | Loss: %.6f" %
                                 (step, current_loss))
    isTrained = True


def predict():
    nu = 2
    omega = 2 * torch.pi * nu


    t = torch.linspace(0, 1, 100).unsqueeze(-1).unsqueeze(0).to(device)
    t.requires_grad = True
    x_pred = model(t.float())

    x_true = x0_true * torch.cos(omega*t)

    fs = 13
    plt.scatter(t[0].cpu().detach().numpy(), x_pred[0].cpu().detach().numpy(), label='pred',
                marker='o',
                alpha=.7,
                s=50)
    plt.plot(t[0].cpu().detach().numpy(), x_true[0].cpu().detach().numpy(),
             color='blue',
             label='analytical')
    plt.xlabel('t', fontsize=fs)
    plt.ylabel('x(t)', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend()
    plt.title('x(t)')
    # plt.savefig('x.png')
    # plt.show()

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

    plt.clf()
    return my_base64_jpgData

async def pinn() :
    global mtemplate
    global isTrained



    base64_encoded_image = b''


    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader,autoescape = False)
    table_templ = env.get_template("index.html")

    if isTrained:
        base64_encoded_image = predict()

    table_templ = table_templ.render(myImage=base64_encoded_image)

    return HTMLResponse(mtemplate('<script></script>', table_templ))


async def train_pinn(background_tasks: BackgroundTasks, N:Optional[int] = None):
    global isTrained
    global isTraining

    if (not isTrained) and (not isTraining):
        background_tasks.add_task(train)
    return {"ok" : isTrained , 'training' : isTraining}

if __name__ == '__main__':

    @asynccontextmanager
    async def lifespan(app: jsonrpc.API):
        yield


    app = jsonrpc.API(lifespan=lifespan)
    app.bind_entrypoint(api_v1)
    app.add_api_route("/pinn", pinn, methods=["GET"])
    app.add_api_route("/train", train_pinn, methods=["GET"])

    import uvicorn
    uvicorn.run(app, host=cur_host, port=8010)