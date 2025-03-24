from contextlib import asynccontextmanager
from fastapi import Body
import json
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi import FastAPI, BackgroundTasks, Request, Depends, WebSocket, Query, Response, APIRouter
from fastapi.websockets import WebSocketState, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import jinja2
import io
import base64

import mTemplate
from mongo_schemas import *
from mNeuralNetService import NeuralNetMicroservice

import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.websocket("/ws_ping")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await neural_net_manager.ws_manager.connect(websocket, glob_user)
    except Exception as err:
        print(f"ws_ping exception {err}")

    while True:
        try:
            data = await websocket.receive_text()
            await asyncio.sleep(2e-3)
        except WebSocketDisconnect:
            print("WebSocketDisconnect {type(websocket)}")
            neural_net_manager.ws_manager.disconnect(websocket, glob_user)
        except Exception as err:
            print(f"cant receive text {err}")
            return


async def get_host():
    return {"cur_host": cur_host}


async def create_neural_model_post(params: Optional[mHyperParams] = None):
    if not params.save_weights_path:
        params.save_weights_path = "weights/osc_1d.pth"

    await neural_net_manager.create_model(params)

    neural_list = await mNeuralNetMongo.get_all()

    table_html = templates.get_template("html/table_template.html").render({"items": neural_list})

    letter = ChatMessage(user=glob_user, msg_type='jinja_tmpl', data=['model_desc', table_html])
    await neural_net_manager.ws_manager.send_personal_message_json(letter)

    return {"resp": "OK"}


async def train_neural_net():
    res = await neural_net_manager.train_model()
    return {"result": res}


async def load_model_handler(model_id: Optional[mNeuralNetMongo] = None):
    res = await neural_net_manager.load_nn(model_id)

    print(res['my_model_desc'])

    return res


async def root(request: Request):
    neural_list = await mNeuralNetMongo.get_all()

    table_html = templates.get_template("html/table_template.html").render({"items": neural_list})
    chat_html = templates.get_template("html/chat_template.html").render({})

    if neural_list:
        # await load_model_handler(neural_list[0].id)

        print(type(neural_list[0]))
        print(neural_list[0])
        # await neural_net_manager.load_nn()

    return templates.TemplateResponse(
        name="html/index.html",
        context={"request": request, "table_html": table_html, "chat_html": chat_html}
    )


async def run_neural_net():
    base64_encoded_image = await neural_net_manager.run_model()

    loader = jinja2.FileSystemLoader("./templates")
    env = jinja2.Environment(loader=loader, autoescape=False)
    table_templ = env.get_template("index.html")

    table_templ = table_templ.render(myImage=base64_encoded_image)

    return HTMLResponse(mTemplate.mtemplate('<script></script>', table_templ, ''))


async def run_neural_net_pict():
    base64_encoded_image = await neural_net_manager.run_model()

    # loader = jinja2.FileSystemLoader("./templates")
    # env = jinja2.Environment(loader=loader, autoescape=False)
    # table_templ = env.get_template("index.html")
    #
    image_html = templates.get_template("/html/chat_template.html").render({"image": base64_encoded_image})
    #
    letter = ChatMessage(user=glob_user, msg_type='jinja_tmpl', data=['chat_container_id', image_html])
    await neural_net_manager.ws_manager.send_personal_message_json(letter)

    return {"OK"}



async def update_train_parameters(params: dict = None):

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
        return {f"error {str(e)}"}


def main(main_loop):
    async def init():
        await init_beanie(database=client[database_name], document_models=[
            MongoEstimate,
            MongoRecord,
            mOptimizerMongo,
            mDataSetMongo,
            mHyperParamsMongo,
            mNeuralNetMongo,
            mWeightMongo
        ])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init()
        yield

    app = FastAPI(lifespan=lifespan)

    app.add_api_route("/create_model", create_neural_model_post, methods=["POST"])
    app.add_api_route("/load_model", load_model_handler, methods=["POST"])
    app.add_api_route("/run", run_neural_net, methods=["GET"])
    app.add_api_route("/run", run_neural_net_pict, methods=["POST"])
    app.add_api_route("/train", train_neural_net, methods=["GET"])
    app.add_api_route("/", root, methods=["GET"])
    app.add_api_route("/update_train_params", update_train_parameters, methods=["POST"])
    app.add_api_route("/clear_database", clear_database, methods=["GET"])
    app.add_api_route("/get_host", get_host, methods=["GET"])

    app.mount("/static", StaticFiles(directory="templates"), name="static")

    app.include_router(router)

    import uvicorn

    config = uvicorn.Config(app, host=cur_host, port=8010, loop=main_loop, access_log=False)
    server = uvicorn.Server(config)
    main_loop.run_until_complete(server.serve())


if __name__ == '__main__':
    client = AsyncIOMotorClient("mongodb://" + database_ulr + ":27017")
    neural_net_manager = NeuralNetMicroservice()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main(loop)
