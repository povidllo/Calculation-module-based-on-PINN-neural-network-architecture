import numpy as np
import torch
import scipy.io
import sys

def calculate_l2_error(path_true_data, model, device, test_data_generator):
    test_data, _, [N, T] = test_data_generator()
    test_variables = torch.FloatTensor(test_data).to(device)
    with torch.no_grad():
        u_pred = model(test_variables)
    u_pred = u_pred.cpu().numpy().reshape(N, T)

    # Сравнение с эталоном
    data = scipy.io.loadmat(sys.path[0] + path_true_data)
    exact_solution = np.real(data['uu'])
    error = np.linalg.norm(u_pred - exact_solution, 2) / np.linalg.norm(exact_solution, 2)
    return error