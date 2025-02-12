from fastapi import FastAPI
from beanie import init_beanie
from raw_database import HyperParam, Weight
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn  # Импортируем uvicorn для запуска приложения
import asyncio

# Инициализация FastAPI
app = FastAPI()

# Функция для инициализации Beanie с MongoDB
async def init():
    client = AsyncIOMotorClient("mongodb://localhost:27017")  # Замените на ваш URI
    database = client['mydatabase']  # Название вашей базы данных
    await init_beanie(database, document_models=[HyperParam, Weight])

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
async def create_weight(weight: Weight):
    await weight.insert()
    return weight

@app.get("/weights/")
async def get_weights():
    weights = await Weight.find_all().to_list()
    return weights

# Добавляем запуск приложения через uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)