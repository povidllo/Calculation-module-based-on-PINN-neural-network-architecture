from typing import List, Dict, Optional, Union, Any
from fastapi import Body, UploadFile
from pydantic import BaseModel, ConfigDict
from beanie import Document, Indexed, init_beanie, Link, WriteRules
import fastapi_jsonrpc as jsonrpc

cur_host = 'localhost'
glob_user = 'andy'
database_ulr = 'localhost'
database_name = "testDB"
pinn_mongo_database = "PINN"
rec_mongo_database = "REC"

opti_mongo_database = "OPTI"
dataset_mongo_database = "DATASET"
hyper_params_mongo_database = "HYPER_PARAMS"
neural_net_mongo_database = "NEURAL_NET"
weights_mongo_database = "WEIGHTS"
class MyError(jsonrpc.BaseError):
    CODE = 5000
    MESSAGE = 'My error'

    class DataModel(BaseModel):
        details: str


class ChatMessage(BaseModel):
    msg_id: Optional[str] = None
    msg_type: Optional[str] = None
    data : Optional[list] = None
    user: Optional[str] = None

class mRecord(BaseModel):
    record: Optional[dict] = None
    tag: Optional[str] = None #loss_graph - data for plot

class mOptimizer(BaseModel):
    method: Optional[str] = None
    params : Optional[dict] = None

class mDataSet(BaseModel):
    params: Optional[dict] = None
    num_dots : Optional[dict] = {
        "train": 400,
        "physics": 50
    }

class mHyperParams(BaseModel):
    my_model_type : Optional[str] = None
    my_model_desc: Optional[str] = None

    input_dim : Optional[int] = 1
    output_dim : Optional[int] = 1
    hidden_sizes : Optional[List[int]] = [32, 32, 32]

    epochs : Optional[int] = 100
    
    Fourier : Optional[bool] = False
    FInputDim : Optional[bool] = None
    FourierScale : Optional[bool] = None
    
    path_true_data : Optional[str] = "/equations/oscillator_eq/OSC.npy"
    save_weights_path : Optional[str] = "/osc_1d.pth"


class mNeuralNet(BaseModel):
    stored_item_id : Optional[str] = None
    status : Optional[str] = None


class MongoRecord(mRecord, Document):
    class Settings:
        name = rec_mongo_database

    class Config:
        arbitrary_types_allowed = True
        # json_encoders = {bytes: lambda s: str(s, errors='ignore') }

class mOptimizerMongo(mOptimizer, Document):
    class Settings:
        name = opti_mongo_database

    class Config:
        arbitrary_types_allowed = True


class mDataSetMongo(mDataSet, Document):
    class Settings:
        name = dataset_mongo_database

    class Config:
        arbitrary_types_allowed = True

class mHyperParamsMongo(mHyperParams, Document):

    class Settings:
        name = hyper_params_mongo_database

    class Config:
        arbitrary_types_allowed = True

class mWeight(BaseModel):
    weights_id: Optional[str] = None
    tag: Optional[str] = None

class mWeightMongo(mWeight, Document):
    class Settings:
        name = weights_mongo_database

    class Config:
        arbitrary_types_allowed = True

class mNeuralNetMongo(mNeuralNet, Document):
    hyper_param : Optional[Link[mHyperParamsMongo]] = None
    data_set : Optional[list[Link[mDataSetMongo]]] = None
    optimizer :  Optional[list[Link[mOptimizerMongo]]] = None
    records: Optional[list[Link[MongoRecord]]] = None
    weights: Optional[Link[mWeightMongo]] = None

    @staticmethod
    async def get_all():
        entries = await mNeuralNetMongo.find_all(fetch_links=True).to_list()
        return entries

    @staticmethod
    async def m_insert(el:Document):
        # await el.save(link_rule=WriteRules.WRITE)
        await el.insert(link_rule=WriteRules.WRITE)

    @staticmethod
    async def get_item_by_id(el:Document):

        res = await mNeuralNetMongo.get(el.stored_item_id)
        return res

    @staticmethod
    async def m_save(el:Document):
        await el.save(link_rule=WriteRules.WRITE)
        # await el.insert(link_rule=WriteRules.WRITE)

    @staticmethod
    async def update_train_params(model_id: str, train_params: mHyperParamsMongo):
        """Обновляет параметры обучения для модели"""
        model = await mNeuralNetMongo.get(model_id)
        model.hyper_param = train_params
        await model.save(link_rule=WriteRules.WRITE)

    class Settings:
        name = neural_net_mongo_database

    class Config:
        arbitrary_types_allowed = True

class mEstimate(BaseModel):
    file_id : Optional[str] = None
    frame_interval: Optional[list[int]] = None
    roi: Optional[list[int]] = None
    records: Optional[list[Link[MongoRecord]]] = None
    tag: Optional[str] = None

class MongoEstimate(mEstimate, Document):
    @staticmethod
    async def m_insert(el:Document):
        await el.save(link_rule=WriteRules.WRITE)

    @staticmethod
    async def find_entries_all():
        all = await MongoEstimate.find({}, fetch_links=True).to_list()
        print(all)
        return all

    class Settings:
        name = pinn_mongo_database

    class Config:
        arbitrary_types_allowed = True
        # json_encoders = {bytes: lambda s: str(s, errors='ignore') }

async def clear_all_collections():
    """Очищает все коллекции в базе данных"""
    try:
        print("Starting database cleanup...")
        
        collections = [
            MongoRecord,
            mOptimizerMongo,
            mDataSetMongo,
            mHyperParamsMongo,
            mNeuralNetMongo,
            MongoEstimate,
            mWeightMongo
        ]
        
        for collection in collections:
            print(f"Clearing {collection.__name__}...")
            try:
                await collection.delete_all()
            except Exception as e:
                print(f"Error clearing {collection.__name__}: {str(e)}")
                # Продолжаем очистку других коллекций
                continue
        
        print("All collections cleared successfully")
    except Exception as e:
        print(f"Error during collection cleanup: {str(e)}")
        raise e