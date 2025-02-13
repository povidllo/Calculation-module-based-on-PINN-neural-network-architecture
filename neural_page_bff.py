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

router = APIRouter()




async def create_neural_model(model_type : Optional[str] = None, inNu : Optional[int] = None, desc : Optional[str] = 'hello world'):
    if (model_type is not None):
        params = mHyperParams(mymodel_type=model_type, mymodel_desc=desc)
        await neural_net_manager.create_model(params)
        if (inNu is not None):
            mdataset = mDataSet(params={'nu':inNu})
            await neural_net_manager.set_dataset(mdataset)

    return {"resp" : "OK"}



async def train_neural_net():
    res = await neural_net_manager.train_model()

    return {"result": res}


async def run_neural_net():
 


    base64_encoded_image = await neural_net_manager.run_model()

    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader, autoescape=False)
    table_templ = env.get_template("index.html")

    table_templ = table_templ.render(myImage=base64_encoded_image)

    return HTMLResponse(mTemplate.mtemplate('<script></script>', table_templ))



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
    app.add_api_route("/train", train_neural_net, methods=["GET"])


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