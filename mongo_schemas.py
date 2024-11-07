from typing import List, Dict, Optional, Union, Any
from fastapi import Body, UploadFile
from pydantic import BaseModel, ConfigDict
from beanie import Document, Indexed, init_beanie, Link, WriteRules
import fastapi_jsonrpc as jsonrpc

database_ulr = 'localhost'
database_name = "testDB"
pinn_mongo_database = "PINN"


class MyError(jsonrpc.BaseError):
    CODE = 5000
    MESSAGE = 'My error'

    class DataModel(BaseModel):
        details: str


class mRecord(BaseModel):
    record: Optional[list] = None

class mongo_Record(mRecord, Document):
    class Settings:
        name = pinn_mongo_database

    class Config:
        arbitrary_types_allowed = True
        # json_encoders = {bytes: lambda s: str(s, errors='ignore') }

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
        ret = await mongo_Estimate.find_all().to_list()
        return ret

    class Settings:
        name = pinn_mongo_database

    class Config:
        arbitrary_types_allowed = True
        # json_encoders = {bytes: lambda s: str(s, errors='ignore') }