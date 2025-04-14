import sys
import os
from pprint import pprint
import numpy as np


def generator():
    '''
    Генерация тестовых данных для вычисления L2 ошибки
    Возвращает:
        test_data: np.array, тестовые данные, готовые для подачи в модель
        []: tuple, массив numpy точек пространства(нужен для визуализации)
        []: tuple, размерность тестовых данных
    '''
    num_t = 201
    num_x = 513
    t = np.linspace(0, 1, num_t).reshape(-1, 1)
    x = np.linspace(-1, 1, num_x)[:-1].reshape(-1, 1)
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T
    X_star = np.tile(x, (1, T))
    t_test = T_star.flatten()[:, None]
    x_test = X_star.flatten()[:, None]
    return np.concatenate((t_test, x_test), axis=1), [t, x], [N, T]