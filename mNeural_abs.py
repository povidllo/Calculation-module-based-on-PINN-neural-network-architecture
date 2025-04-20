from mongo_schemas import *
import abc
from pymongo import MongoClient
import gridfs
import pickle
from bson.objectid import ObjectId

from pinn_init_torch import pinn


class AbsNeuralNet(abc.ABC):
    neural_model : mNeuralNetMongo = None

    client = MongoClient("mongodb://localhost")
    db = client.testDB
    fs = gridfs.GridFS(db)

    async def create_model(self, params : mHyperParams):
        # Создаем гиперпараметры и сохраняем их
        hyper_params = mHyperParamsMongo(**params.model_dump())
        await hyper_params.insert()
        
        # Создаем модель с сохраненными гиперпараметрами
        self.neural_model = mNeuralNetMongo(hyper_param=hyper_params)
        self.neural_model.records = []

        
        # Если есть параметры оптимизатора, создаем его
        if hasattr(params, 'optimizer'):
            optimizer = mOptimizerMongo(
                method=params.optimizer.method,
                params=params.optimizer.params
            )
            await optimizer.insert()
            self.neural_model.optimizer = [optimizer]
        
        # Сохраняем модель
        await mNeuralNetMongo.m_insert(self.neural_model)
        
        # Создаем и сохраняем датасет
        await self.set_dataset()

    async def abs_save_plot(self, plot):
        new_rec = MongoRecord(record={'raw': plot})
        await self.append_rec_to_nn(new_rec)

    async def abs_save_weights(self, obj):
        try:
            items_dumps = pickle.dumps(obj, protocol=2)
            ret_id = self.fs.put(items_dumps, filename='my_weights')
            
            weight = mWeightMongo(weights_id=str(ret_id), tag='weight')
            await weight.insert()
            
            self.neural_model.weights = weight
            await mNeuralNetMongo.m_save(self.neural_model)

            print('gridfs id', ret_id)
        except Exception as exp:
            print('save_weights_exp', exp)

    async def abs_set_optimizer(self):
        try:
            opti = mOptimizerMongo(method='Adam', params={'lr': 0.001})
            print('Создан новый оптимизатор:', opti)

            await opti.insert()
            self.neural_model.optimizer = [opti]
            await mNeuralNetMongo.m_save(self.neural_model)
        except Exception as exp:
            print('save_optimizer_exp', exp)

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


    async def append_rec_to_nn(self, new_rec : MongoRecord):
        self.neural_model.records.append(new_rec)

        await mNeuralNetMongo.m_save(self.neural_model)

    async def update_dataset_for_nn(self, new_dataset : mDataSetMongo):
        await self.neural_model.data_set[0].delete()
        self.neural_model.data_set = [new_dataset]

        await mNeuralNetMongo.m_save(self.neural_model)

    '''
    загружает из БД выбранную в интерфейсе модель
    '''
    async def load_model(self, in_model: mNeuralNet, in_device):
        load_nn = await mNeuralNetMongo.get(in_model.stored_item_id, fetch_links=True)

        self.neural_model = load_nn
        self.neural_model.records = []
        await self.set_dataset()

        self.mydevice = in_device
        self.mymodel = pinn(self.neural_model.hyper_param).to(self.mydevice)
        await self.set_optimizer()

        cur_state = await self.abs_load_weights()
        if (cur_state is not None):
            self.mymodel.load_state_dict(cur_state)
        else:
            print("No saved weights found, using initialized weights")

    async def update_train_params(self, train_params: dict = None):
        # Обновляем количество эпох
        self.neural_model.hyper_param.epochs = train_params.get('epochs')
        
        # Если оптимизатор уже существует, обновляем его
        if self.neural_model.optimizer:
            print('Обновляем существующий оптимизатор')
            optimizer = self.neural_model.optimizer[0]
            optimizer.method = train_params.get('method')
            optimizer.params = train_params.get('params', {'lr': 0.001})

            # optimizer.params = {''}
            await optimizer.save()
        else:
            # Создаем новый оптимизатор только если его нет
            print('Создаем новый оптимизатор')
            optimizer = mOptimizerMongo(
                method=train_params.get('method'),
                params=train_params.get('params', {'lr': 0.001})
            )
            await optimizer.insert()
            self.neural_model.optimizer = [optimizer]
        
        await mNeuralNetMongo.m_save(self.neural_model)
        
        # Пересоздаем оптимизатор в PyTorch
        await self.set_optimizer(optimizer)

    @abc.abstractmethod
    async def construct_model(self, params : mHyperParams, in_device): pass

    @abc.abstractmethod
    def set_optimizer(self, opti : mOptimizer = None): pass

    @abc.abstractmethod
    async def set_dataset(self, dataset: mDataSet = None): pass

    '''
    тренировка модели
    '''
    @abc.abstractmethod
    async def train(self): pass

    '''
    обученная модель рассчитывает значения и возвращает график
    '''
    @abc.abstractmethod
    async def calc(self): pass