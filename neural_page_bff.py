from contextlib import asynccontextmanager
from fastapi import Body
import json
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi import FastAPI, BackgroundTasks, Request, Depends,WebSocket, Query, Response, APIRouter
from fastapi.websockets import WebSocketState, WebSocketDisconnect

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

router = APIRouter()






@router.websocket("/ws_ping")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await neural_net_manager.ws_manager.connect(websocket, glob_user)
    except Exception as err:
        print('ws_ping exception', err)

    while True:
        try:
            data = await websocket.receive_text()
            await asyncio.sleep(2e-3)
        except WebSocketDisconnect:
            print('WebSocketDisconnect ' , type(websocket))
            neural_net_manager.ws_manager.disconnect(websocket, glob_user)
        except Exception as err:
            print('cant recive text', err)
            return

# async def create_neural_model(model_type : Optional[str] = None, inNu : Optional[int] = None, desc : Optional[str] = 'hello world'):
#     if (model_type is not None):
#         params = mHyperParams(mymodel_type=model_type, mymodel_desc=desc)
#         await neural_net_manager.create_model(params)
#         if (inNu is not None):
#             mdataset = mDataSet(params={'nu':inNu})
#             await neural_net_manager.set_dataset(mdataset)
#
#     return {"resp" : "OK"}

async def create_neural_model_post(params : Optional[mHyperParams] = None):
    # Если путь к весам не указан, используем значение по умолчанию
    if not params.save_weights_path:
        params.save_weights_path = "weights/osc_1d.pth"
        
    await neural_net_manager.create_model(params)

    neural_list = await mNeuralNet_mongo.get_all()
    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader, autoescape = False)
    neural_table_templ = env.get_template("table_template.html")
    neural_table_templ = neural_table_templ.render(items=neural_list)

    letter = chatMessage(user=glob_user, msg_type='jinja_tmpl', data=['neural_table', neural_table_templ])
    await neural_net_manager.ws_manager.send_personal_message_json(letter)

    return {"resp" : "OK"}

async def train_neural_net(params : Optional[mHyperParams] = None):
    # Теперь params не используется, так как берутся из базы
    res = await neural_net_manager.train_model()
    return {"result": res}

async def load_model_handler(inp_nn : Optional[mNeuralNet] = None):

    res = await neural_net_manager.load_nn(inp_nn)


    return res


async def root():
    base64_encoded_image = b''
    neural_list = []

    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader,autoescape = False)

    img_view_templ = env.get_template("img_tmpl.html")
    neural_table_templ = env.get_template("table_template.html")

    neural_list = await mNeuralNet_mongo.get_all()
    # print('entries', mnl)


    img_view_templ = img_view_templ.render(myImage=base64_encoded_image)
    neural_table_templ = neural_table_templ.render(items=neural_list)

    return HTMLResponse(mTemplate.mtemplate('<script></script>', neural_table_templ, img_view_templ))

async def run_neural_net():
 

    base64_encoded_image = await neural_net_manager.run_model()

    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader, autoescape=False)
    table_templ = env.get_template("index.html")

    table_templ = table_templ.render(myImage=base64_encoded_image)

    return HTMLResponse(mTemplate.mtemplate('<script></script>', table_templ, ''))


async def run_neural_net_pict():
    base64_encoded_image = await neural_net_manager.run_model()

    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader, autoescape=False)
    table_templ = env.get_template("index.html")

    table_templ = table_templ.render(myImage=base64_encoded_image)

    letter = chatMessage(user=glob_user, msg_type='jinja_tmpl', data=['my_img', table_templ])
    await neural_net_manager.ws_manager.send_personal_message_json(letter)
    
    return {"OK"}

async def update_train_parameters(params : Optional[mHyperParams] = None):
    if neural_net_manager.inner_model is not None:
        await neural_net_manager.inner_model.update_train_params(params)
        return {"result": "OK"}
    return {"result": "No model loaded"}

async def clear_database():
    try:
        print('Starting database cleanup...')
        await clear_all_collections()
        print('Collections cleared')
        
        # Обновляем таблицу после очистки через перезагрузку страницы
        return RedirectResponse(url="/")
        
    except Exception as e:
        print(f"Error clearing database: {str(e)}")
        return {"error": str(e)}

def main(loop):

    async def init():
        await init_beanie(database=client[database_name], document_models=[
            mongo_Estimate, 
            mongo_Record, 
            mOptimizer_mongo, 
            mDataSet_mongo, 
            mHyperParams_mongo, 
            mNeuralNet_mongo,
            mWeight_mongo  # Добавляем новую модель для весов
        ])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init()
        yield

    app = FastAPI(lifespan=lifespan)

    # app.add_api_route("/create_model", create_neural_model, methods=["GET"])
    app.add_api_route("/create_model", create_neural_model_post, methods=["POST"])
    app.add_api_route("/load_model", load_model_handler, methods=["POST"])
    app.add_api_route("/run", run_neural_net, methods=["GET"])
    app.add_api_route("/run", run_neural_net_pict, methods=["POST"])
    app.add_api_route("/train", train_neural_net, methods=["GET"])
    app.add_api_route("/", root, methods=["GET"])
    app.add_api_route("/update_train_params", update_train_parameters, methods=["POST"])
    app.add_api_route("/clear_database", clear_database, methods=["GET"])


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