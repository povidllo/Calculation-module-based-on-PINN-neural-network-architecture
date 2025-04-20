import numpy as np

import torch
def generator():
    '''
    Генерация тестовых данных для вычисления L2 ошибки
    Возвращает:
        test_data: np.array, тестовые данные, готовые для подачи в модель
        []: tuple, массив numpy точек пространства(нужен для визуализации)
        []: tuple, размерность тестовых данных
    '''
    num_t = 400
    t = np.linspace(0, 1, num_t).reshape(-1, 1)
    t = torch.FloatTensor(t)
    return t, [t], [num_t]