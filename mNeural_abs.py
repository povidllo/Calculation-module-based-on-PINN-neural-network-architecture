from mongo_schemas import *
import abc


class abs_neural_net(abc.ABC):
    neural_model : mNeuralNet_mongo = None
    

    async def createModel(self, params : mHyperParams):
        self.neural_model = mNeuralNet_mongo(hyper_param=mHyperParams_mongo(**params.model_dump()))
        self.neural_model.records = []
        await self.set_dataset()

        await mNeuralNet_mongo.m_insert(self.neural_model)

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
    def train(self): pass

    @abc.abstractmethod
    async def calc(self): pass