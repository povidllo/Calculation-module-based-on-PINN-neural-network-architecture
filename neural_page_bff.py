from contextlib import asynccontextmanager
from fastapi import (
    FastAPI, APIRouter, WebSocket, WebSocketDisconnect, Request,
    Depends, BackgroundTasks, Query, Response, Body
)
from fastapi.responses import (
    HTMLResponse, RedirectResponse, JSONResponse, FileResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from motor.motor_asyncio import AsyncIOMotorClient

import json
import asyncio
import jinja2

from mongo_schemas import *
from mNeuralNetService import NeuralNetMicroservice

from optimizer_setings import optimizer_dict

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.websocket("/ws_ping")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await neural_net_manager.ws_manager.connect(websocket, glob_user)
        while True:
            await asyncio.sleep(0.002)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        print("WebSocketDisconnect", type(websocket))
        neural_net_manager.ws_manager.disconnect(websocket, glob_user)
    except Exception as err:
        print(f"WebSocket error: {err}")


async def get_host():
    return {"cur_host": cur_host}


@router.post("/create_model")
async def create_neural_model_post(params: mHyperParams = Body(...)):
    params.save_weights_path = params.save_weights_path or "weights/osc_1d.pth"
    await neural_net_manager.create_model(params)
    neural_list = await mNeuralNetMongo.get_all()

    table_html = templates.get_template("html/table_template.html").render({"items": neural_list})
    letter = ChatMessage(user=glob_user, msg_type='jinja_tmpl', data=['model_desc', table_html])
    await neural_net_manager.ws_manager.send_personal_message_json(letter)

    return {"resp": "OK"}


@router.get("/train")
async def train_neural_net():
    result = await neural_net_manager.train_model()
    return {"result": result}


@router.post("/load_model")
async def load_model_handler(model_id: mNeuralNetMongo = Body(...)):
    hyper_param_dict = await neural_net_manager.load_nn(model_id)
    return hyper_param_dict


@router.get("/")
async def root(request: Request):
    neural_list = await mNeuralNetMongo.get_all()

    table_html = templates.get_template("html/table_template.html").render({"items": neural_list})
    chat_html = templates.get_template("html/chat_template.html").render({})
    equation_type_html = templates.get_template("html/equation_template.html").render({"equation_dict": neural_net_manager.models_list})
    hyperparams_html = templates.get_template("html/hyperparams_template.html").render({})

    return templates.TemplateResponse(
        name="html/index.html",
        context={
            "request": request,
            "table_html": table_html,
            "chat_html": chat_html,
            "equation_type_html": equation_type_html,
            "hyperparams_html": hyperparams_html
        }
    )


@router.post("/run")
async def run_neural_net_pict():
    encoded_image = await neural_net_manager.run_model()
    image_html = templates.get_template("html/chat_template.html").render({"image": encoded_image})

    letter = ChatMessage(user=glob_user, msg_type='jinja_tmpl', data=['chat_container_id', image_html])
    await neural_net_manager.ws_manager.send_personal_message_json(letter)

    return {"status": "OK"}


@router.post("/update_train_params")
async def update_train_parameters(params: dict = Body(...)):
    if neural_net_manager.inner_model:
        await neural_net_manager.inner_model.update_train_params(params)
        return {"result": "OK"}
    return {"result": "No model loaded"}


@router.get("/clear_database")
async def clear_database():
    try:
        print('Starting database cleanup...')
        await clear_all_collections()
        print('Collections cleared')
        return RedirectResponse(url="/")
    except Exception as e:
        print(f"Error clearing database: {str(e)}")
        return {"error": str(e)}


@router.post("/update_optimizer_params")
async def update_optimizer_params(optimizer_name: dict = Body(...)):

    opti_name = optimizer_name['optimizer_name']

    optimizer_params_html = (templates.get_template(f"html/optimizer_template.html").render({"optimizer_list": optimizer_dict[opti_name]}))

    letter = ChatMessage(user=glob_user, msg_type='jinja_tmpl', data=['optimizer-params', optimizer_params_html])
    await neural_net_manager.ws_manager.send_personal_message_json(letter)

    opti_id_list = [i['id'] for i in optimizer_dict[opti_name]]


    return {"list_of_id": opti_id_list}


@router.get("/test_chat")
async def test_chat(request: Request):
    result = await neural_net_manager.get_chat()


def main(loop):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_beanie(database=client[database_name], document_models=[
            MongoEstimate, MongoRecord, mOptimizerMongo, mDataSetMongo,
            mHyperParamsMongo, mNeuralNetMongo, mWeightMongo
        ])
        yield

    app = FastAPI(lifespan=lifespan)

    app.include_router(router)
    app.mount("/static", StaticFiles(directory="templates"), name="static")

    import uvicorn
    config = uvicorn.Config(app, host=cur_host, port=8010, loop=loop, access_log=False)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())


if __name__ == '__main__':
    client = AsyncIOMotorClient(f"mongodb://{database_ulr}:27017")
    neural_net_manager = NeuralNetMicroservice()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main(loop)
