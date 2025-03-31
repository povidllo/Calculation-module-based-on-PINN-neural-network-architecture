from mongo_schemas import *
import base64
import io
import numpy as np

class MongoDBHandler:
    @staticmethod
    async def load_model(in_model):
        return await mNeuralNetMongo.get(in_model.stored_item_id, fetch_links=True)

    @staticmethod
    async def save_model_weights(model, weights):
        await model.abs_save_weights(weights)
        print('Model weights saved successfully')

    @staticmethod
    async def save_dataset(dataset):
        await dataset.insert()
        print('Dataset saved successfully')

    @staticmethod
    async def update_model_dataset(neural_model, dataset):
        neural_model.data_set = [dataset]
        await mNeuralNetMongo.m_save(neural_model)
        print('Model dataset updated successfully')

    @staticmethod
    def load_data_from_params(params):
        if 'points_data' not in params:
            return None
        decoded_data = base64.b64decode(params['points_data'])
        buffer = io.BytesIO(decoded_data)
        data = np.load(buffer)
        return {
            'x': torch.FloatTensor(data['x']),
            'x_data': torch.FloatTensor(data['x_data']),
            'x_physics': torch.FloatTensor(data['x_physics']).requires_grad_(True),
            'y': torch.FloatTensor(data['y']),
            'y_data': torch.FloatTensor(data['y_data'])
        }
