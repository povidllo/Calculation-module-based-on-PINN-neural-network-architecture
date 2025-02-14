from typing import List, Dict, Optional, Union, Any
from fastapi import Body, UploadFile
from pydantic import BaseModel, ConfigDict
from beanie import Document, Indexed, init_beanie, Link, WriteRules
import fastapi_jsonrpc as jsonrpc
from collections.abc import Callable, Awaitable
import abc

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

class MyError(jsonrpc.BaseError):
    CODE = 5000
    MESSAGE = 'My error'

    class DataModel(BaseModel):
        details: str


class chatMessage(BaseModel):
    msg_id: Optional[str] = None
    msg_type: Optional[str] = None
    data : Optional[list] = None
    user: Optional[str] = None

class mRecord(BaseModel):
    record: Optional[dict] = None

class mOptimizer(BaseModel):
    method: Optional[str] = None
    params : Optional[dict] = None

class mDataSet(BaseModel):
    power_time_vector : Optional[int] = 100
    params: Optional[dict] = None

class mHyperParams(BaseModel):
    mymodel_type : Optional[str] = None
    mymodel_desc: Optional[str] = None

    hidden_size : Optional[int] = 20
    power_time_vector : Optional[int] = 100
    epochs : Optional[int] = 50
    
class mNeuralNet(BaseModel):
    stored_item_id : Optional[str] = None
    status : Optional[str] = None


class mongo_Record(mRecord, Document):
    class Settings:
        name = rec_mongo_database

    class Config:
        arbitrary_types_allowed = True
        # json_encoders = {bytes: lambda s: str(s, errors='ignore') }

class mOptimizer_mongo(mOptimizer,Document):
    class Settings:
        name = opti_mongo_database

    class Config:
        arbitrary_types_allowed = True


class mDataSet_mongo(mDataSet, Document):
    class Settings:
        name = dataset_mongo_database

    class Config:
        arbitrary_types_allowed = True

class mHyperParams_mongo(mHyperParams, Document):

    class Settings:
        name = hyper_params_mongo_database

    class Config:
        arbitrary_types_allowed = True

class mNeuralNet_mongo(mNeuralNet, Document):
    hyper_param : Optional[Link[mHyperParams_mongo]] = None
    data_set : Optional[list[Link[mDataSet_mongo]]] = None
    optimizer :  Optional[list[Link[mOptimizer_mongo]]] = None
    records: Optional[list[Link[mongo_Record]]] = None



    @staticmethod
    async def get_all():
        entries = await mNeuralNet_mongo.find_all(fetch_links=True).to_list()
        return entries

    @staticmethod
    async def m_insert(el:Document):
        # await el.save(link_rule=WriteRules.WRITE)
        await el.insert(link_rule=WriteRules.WRITE)

    @staticmethod
    async def m_save(el:Document):
        await el.save(link_rule=WriteRules.WRITE)
        # await el.insert(link_rule=WriteRules.WRITE)

    class Settings:
        name = neural_net_mongo_database

    class Config:
        arbitrary_types_allowed = True








class mEstimate(BaseModel):
    file_id : Optional[str] = None
    frame_interval: Optional[list[int]] = None
    roi: Optional[list[int]] = None
    records: Optional[list[Link[mongo_Record]]] = None
    tag: Optional[str] = None

class mongo_Estimate(mEstimate, Document):
    @staticmethod
    async def m_insert(el:Document):
        await el.save(link_rule=WriteRules.WRITE)

    @staticmethod
    async def find_entries_all():
        all = await mongo_Estimate.find( {}, fetch_links=True).to_list()
        print(all)
        return all

    class Settings:
        name = pinn_mongo_database

    class Config:
        arbitrary_types_allowed = True
        # json_encoders = {bytes: lambda s: str(s, errors='ignore') }