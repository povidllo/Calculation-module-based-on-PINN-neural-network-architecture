import numpy as np
import torch
import scipy.io
import sys

def calculate_l2_error(path_true_data, model, device, test_data_generator):
    x, _, _ = test_data_generator()
    y = np.load(sys.path[0] + path_true_data)
    u_pred = model(x)
    u_pred = u_pred.cpu().detach().numpy()
    true= y
    # Сравнение с эталоном
    error = np.linalg.norm(u_pred - true, 2) / np.linalg.norm(true, 2)
    return error