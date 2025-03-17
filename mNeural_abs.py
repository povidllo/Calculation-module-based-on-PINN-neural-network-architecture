from mongo_schemas import *
import abc

from pymongo import MongoClient
import gridfs
import pickle
import numpy as np
from bson.objectid import ObjectId


class abs_neural_net(abc.ABC):
    neural_model : mNeuralNet_mongo = None
    
    client = MongoClient("mongodb://localhost")
    db = client.testDB
    fs = gridfs.GridFS(db)

    async def createModel(self, params : mHyperParams):
        # Создаем гиперпараметры и сохраняем их
        hyper_params = mHyperParams_mongo(**params.model_dump())
        await hyper_params.insert()
        
        # Создаем модель с сохраненными гиперпараметрами
        self.neural_model = mNeuralNet_mongo(hyper_param=hyper_params)
        self.neural_model.records = []
        
        # Если есть параметры оптимизатора, создаем его
        if hasattr(params, 'optimizer'):
            optimizer = mOptimizer_mongo(
                method=params.optimizer.method,
                params=params.optimizer.params
            )
            await optimizer.insert()
            self.neural_model.optimizer = [optimizer]
        
        # Сохраняем модель
        await mNeuralNet_mongo.m_insert(self.neural_model)
        
        # Создаем и сохраняем датасет
        await self.set_dataset()

    async def abs_save_weights(self, obj):
        try:
            items_dumps = pickle.dumps(obj, protocol=2)
            retId = self.fs.put(items_dumps, filename='my_weights')
            
            weight = mWeight_mongo(weights_id=str(retId), tag='weight')
            await weight.insert()
            
            self.neural_model.weights = weight
            await mNeuralNet_mongo.m_save(self.neural_model)

            print('gridfs id', retId)
        except Exception as exp:
            print('save_weights_exp', exp)

    async def abs_load_weights(self):
        weights = None
        if self.neural_model.weights is not None:
            try:
                weights_id = self.neural_model.weights.weights_id
                last_fs_weights = self.fs.find_one({'_id': ObjectId(weights_id)})
                
                if last_fs_weights is not None:
                    weight_dump = last_fs_weights.read()
                    weights = pickle.loads(weight_dump)
                    print('weights loaded', type(weights))
            except Exception as e:
                print('load_weights_exp', e)
        
        return weights


    async def append_rec_to_nn(self, new_rec : mongo_Record):
        self.neural_model.records.append(new_rec)

        await mNeuralNet_mongo.m_save(self.neural_model)

    async def update_dataset_for_nn(self, new_dataset : mDataSet_mongo):
        await self.neural_model.data_set[0].delete()
        self.neural_model.data_set = [new_dataset]

        await mNeuralNet_mongo.m_save(self.neural_model)

    async def update_train_params(self, train_params: dict):
        # Обновляем количество эпох
        self.neural_model.hyper_param.epochs = train_params.get('epochs', 100)
        
        # Если оптимизатор уже существует, обновляем его
        if self.neural_model.optimizer:
            print('Обновляем существующий оптимизатор')
            optimizer = self.neural_model.optimizer[0]
            optimizer.method = train_params.get('method', 'Adam')
            optimizer.params = train_params.get('params', {'lr': 0.001})
            await optimizer.save()
        else:
            # Создаем новый оптимизатор только если его нет
            print('Создаем новый оптимизатор')
            optimizer = mOptimizer_mongo(
                method=train_params.get('method', 'Adam'),
                params=train_params.get('params', {'lr': 0.001})
            )
            await optimizer.insert()
            self.neural_model.optimizer = [optimizer]
        
        await mNeuralNet_mongo.m_save(self.neural_model)
        
        # Пересоздаем оптимизатор в PyTorch
        self.set_optimizer(optimizer)

    @abc.abstractmethod
    async def construct_model(self, params : mHyperParams, in_device): pass

    @abc.abstractmethod
    def set_optimizer(self, opti : mOptimizer = None): pass

    @abc.abstractmethod
    async def set_dataset(self): pass

    @abc.abstractmethod
    async def load_model(self, in_model : mNeuralNet, in_device): pass

    @abc.abstractmethod
    async def train(self): pass

    @abc.abstractmethod
    async def calc(self): pass