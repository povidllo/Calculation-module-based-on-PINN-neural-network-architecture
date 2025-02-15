import sys
import os
from pprint import pprint
import numpy as np

# Добавляем родительскую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cfg_test_data_gen import get_config

def generator():
    '''
    Генерация тестовых данных для вычисления L2 ошибки
    Возвращает:
        test_data: np.array, тестовые данные, готовые для подачи в модель
        []: tuple, массив numpy точек пространства(нужен для визуализации)
        []: tuple, размерность тестовых данных
    '''
    cfg = get_config()
    num_t = cfg.num_dots[0]
    num_x = cfg.num_dots[1]
    t = np.linspace(0, 1, num_t).reshape(-1, 1)
    x = np.linspace(-1, 1, num_x)[:-1].reshape(-1, 1)
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T
    X_star = np.tile(x, (1, T))
    t_test = T_star.flatten()[:, None]
    x_test = X_star.flatten()[:, None]
    return np.concatenate((t_test, x_test), axis=1), [t, x], [N, T]