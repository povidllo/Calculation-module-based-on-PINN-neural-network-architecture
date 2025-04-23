import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
import sys
def vizualize(path_true_data, model, device, test_data_generator):
    # model.load_state_dict(torch.load(sys.path[0] + path_true_data, map_location=torch.device(device)))

    x, _, _ = test_data_generator()
    y = np.load(sys.path[0] + path_true_data)
    u_pred = model(x)
    u_pred = u_pred.cpu().detach().numpy()
    true= y

    plt.figure(figsize=(10, 6))
    
    # Преобразуем x в numpy array если это тензор
    x_plot = x.cpu().numpy() if torch.is_tensor(x) else x
    
    # Строим оба графика
    plt.plot(x_plot, true, 'b-', linewidth=2, label='Истинное решение')
    plt.plot(x_plot, u_pred, 'r--', linewidth=2, label='Предсказание модели')
    
    # Настройки графика
    plt.xlabel('Временная координата (t)', fontsize=12)
    plt.ylabel('Значение y', fontsize=12)
    plt.title('Сравнение предсказаний модели с эталонным решением', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Рассчитываем и выводим ошибку
    error = np.linalg.norm(u_pred - true, 2) / np.linalg.norm(true, 2)
    plt.text(0.05, 0.95, f'Средняя абсолютная ошибка: {error:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()