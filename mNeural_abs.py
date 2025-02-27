from mongo_schemas import *
import abc

from pymongo import MongoClient
import gridfs
import pickle
import numpy as np


class abs_neural_net(abc.ABC):
    neural_model : mNeuralNet_mongo = None
    
    client = MongoClient("mongodb://localhost")
    db = client.testDB
    fs = gridfs.GridFS(db)

    async def createModel(self, params : mHyperParams):
        self.neural_model = mNeuralNet_mongo(hyper_param=mHyperParams_mongo(**params.model_dump()))
        self.neural_model.records = []
        await self.set_dataset()

        await mNeuralNet_mongo.m_insert(self.neural_model)
        
        
    async def abs_save_weights(self, obj):
        try:
            # """Сохранение модели"""
            # weights = self.mymodel.state_dict()
            # torch.save(weights, path)

            # state_dict = self.mymodel.state_dict()
            # weights_list = {key: tensor.tolist() for key, tensor in state_dict.items()}
            # print('self.mymodel.state_dict()', type(weights_list), weights_list)

            # item = mongo_Record(record={'weights': weights_list})
            items_dumps = pickle.dumps(obj, protocol=2)
            # item_base64 = base64.b64encode(items_dumps).decode()


            retId = self.fs.put(items_dumps, filename='my_weights')
            if (self.neural_model.records is None):
                self.neural_model.records = []

            item = mongo_Record(record={'tag' : 'weight', 'weights_id': retId})
            self.neural_model.records.append(item)
            await mNeuralNet_mongo.m_save(self.neural_model)


            print('gridfs id', retId)
        except Exception as exp:
            print('save_weights_exp', exp)

    async def abs_load_weights(self):
        weights = None
        last_fs_weights = None
        if (self.neural_model.records is not None):
            for it in self.neural_model.records:
                if ('tag' in it.record):
                    if (it.record['tag'] == 'weight'):
                        print('weights_id', it.record['weights_id'])
                        last_fs_weights = self.fs.find_one({'_id': it.record['weights_id']})


        if (last_fs_weights is not None):
            weight_dump = last_fs_weights.read()
            weights = pickle.loads(weight_dump)
            print('weights', type(weights))

        return weights


    async def append_rec_to_nn(self, new_rec : mongo_Record):
        self.neural_model.records.append(new_rec)

        await mNeuralNet_mongo.m_save(self.neural_model)

    async def update_dataset_for_nn(self, new_dataset : mDataSet_mongo):
        await self.neural_model.data_set[0].delete()
        self.neural_model.data_set = [new_dataset]

        await mNeuralNet_mongo.m_save(self.neural_model)

    async def update_train_params(self, train_params : mHyperParams):
        # Обновляем параметры обучения в модели
        self.neural_model.hyper_param.epochs = train_params.epochs
        self.neural_model.hyper_param.optimizer = train_params.optimizer
        self.neural_model.hyper_param.optimizer_lr = train_params.optimizer_lr
        
        # Сохраняем изменения в базе данных
        await mNeuralNet_mongo.m_save(self.neural_model)

    @abc.abstractmethod
    async def construct_model(self, params : mHyperParams, in_device): pass

    @abc.abstractmethod
    def set_optimizer(self, opti : mOptimizer = None): pass

    @abc.abstractmethod
    async def set_dataset(self, dataset : mDataSet = None): pass

    @abc.abstractmethod
    async def load_model(self, in_model : mNeuralNet, in_device): pass

    @abc.abstractmethod
    async def train(self): pass

    @abc.abstractmethod
    async def calc(self): pass