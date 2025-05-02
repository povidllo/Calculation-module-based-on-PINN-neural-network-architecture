import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
import sys

def vizualize(path_true_data, model, device, test_data_generator):
    test_data, [t, x], [N, T] = test_data_generator()
    test_variables = torch.FloatTensor(test_data).to(device)
    with torch.no_grad():
        u_pred = model(test_variables)
    u_pred = u_pred.cpu().numpy().reshape(N,T)

    data = scipy.io.loadmat(sys.path[0] + path_true_data)
    Exact = np.real(data['uu'])
    err = u_pred-Exact

    err = np.linalg.norm(err,2)/np.linalg.norm(Exact,2)
    print(f"L2 Relative Error: {err}")

    # plt.figure(figsize=(10, 5))
    # plt.imshow(u_pred, interpolation='nearest', cmap='jet',
    #             extent=[t.min(), t.max(), x.min(), x.max()],
    #             origin='lower', aspect='auto')
    # plt.clim(-1, 1)
    # plt.ylim(-1,1)
    # plt.xlim(0,1)
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.title('u(t,x)')
    # plt.colorbar()  # Добавим цветовую шкалу для наглядности
    # plt.show()


    # Создаем сетку для 3D графиков
    T_mesh, X_mesh = np.meshgrid(t, x)
    
    # 3D визуализация предсказанного решения
    fig = plt.figure(figsize=(18, 6))
    
    # Первый субплот - предсказанное решение
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(T_mesh, X_mesh, u_pred, cmap='jet',
                            linewidth=0, antialiased=True)
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('Время (t)')
    ax1.set_ylabel('Пространство (x)')
    ax1.set_zlabel('u(t,x)')
    ax1.set_title('Предсказанное решение')
    ax1.set_zlim(-1, 1)

    # Второй субплот - истинное решение
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(T_mesh, X_mesh, Exact, cmap='jet',
                            linewidth=0, antialiased=True)
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    ax2.set_xlabel('Время (t)')
    ax2.set_ylabel('Пространство (x)')
    ax2.set_zlabel('u(t,x)')
    ax2.set_title('Истинное решение')
    ax2.set_zlim(-1, 1)

    plt.tight_layout()
    plt.show()