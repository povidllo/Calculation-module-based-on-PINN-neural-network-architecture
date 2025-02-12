from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from beanie import init_beanie
from raw_database import HyperParam, Weights, NeuralNetwork
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn  # Импортируем uvicorn для запуска приложения
import asyncio
from Modules.new_main import Trainable

# Инициализация FastAPI
app = FastAPI()

# Функция для инициализации Beanie с MongoDB
async def init():
    client = AsyncIOMotorClient("mongodb://localhost:27017")  # Замените на ваш URI
    database = client['mydatabase']  # Название вашей базы данных
    await init_beanie(database, document_models=[HyperParam, Weights, NeuralNetwork])

# Событие запуска приложения
@app.on_event("startup")
async def app_init():
    await init()

# Пример эндпоинта для добавления данных
@app.post("/hyperparams/")
async def create_hyperparam(hyperparam: HyperParam):
    await hyperparam.insert()
    return hyperparam

@app.get("/hyperparams/")
async def get_hyperparams():
    hyperparams = await HyperParam.find_all().to_list()
    return hyperparams

@app.post("/weights/")
async def create_weight(weight: Weights):
    await weight.insert()
    return weight

@app.get("/weights/")
async def get_weights():
    weights = await Weights.find_all().to_list()
    return weights

@app.post("/NeuralNetwork/")
async def create_neural_network(neural_network: NeuralNetwork):
    await neural_network.insert()
    return neural_network

@app.get("/NeuralNetwork/")
async def get_neural_network():
    neural_network = await NeuralNetwork.find_all().to_list()
    return neural_network

global model
@app.post("/NeuralNetwork/Train")
async def train_neural_network():
    global model
    model = Trainable("oscillator")
    print(model.equation)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, model.train)


@app.get("/NeuralNetwork/Inference")
async def inference_neural_network():
    global model
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, model.inference)

# Добавляем запуск приложения через uvicorn
if __name__ == "__main__":
    uvicorn.run("raw_back:app", host="127.0.0.1", port=8000, reload=True)