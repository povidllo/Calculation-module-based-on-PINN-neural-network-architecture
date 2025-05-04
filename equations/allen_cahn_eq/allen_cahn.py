import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from mongo_schemas import *
from mNeural_abs import *

import io
import base64
import scipy.io

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

from pinn_init_torch import pinn
from equations.allen_cahn_eq.test_data_generator import generator as test_data_generator

# torch.manual_seed(123)
# np.random.seed(44)
# torch.cuda.manual_seed(44)

class allen_cahn_nn(AbsNeuralNet):
    mymodel = None
    mydevice = None
    myoptimizer = None

    loss_history = []
    l2_history = []
    best_loss = float('inf')
    best_epoch = 0


    class mySpecialDataSet(AbsNeuralNet.AbsDataSet):
        def equation(self, args):
            u, tx = args.values()
            u_tx = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
            u_t = u_tx[:, 0:1]
            u_x = u_tx[:, 1:2]
            u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
            e = u_t - 0.0001 * u_xx + 5 * u ** 3 - 5 * u
            return e

        def ac_generator(self, num_t, num_x, typ='train'):
            N_f = num_t * num_x
            t = np.linspace(0, 1, num_t).reshape(-1, 1)  # T x 1
            x = np.linspace(-1, 1, num_x).reshape(-1, 1)  # N x 1
            T = t.shape[0]
            N = x.shape[0]
            T_star = np.tile(t, (1, N)).T  # N x T
            X_star = np.tile(x, (1, T))  # N x T

            # Initial condition and boundary condition
            u = np.zeros((N, T))  # N x T
            u[:, 0:1] = (x ** 2) * np.cos(np.pi * x)
            u[0, :] = -np.ones(T)
            u[-1, :] = u[0, :]

            t_data = T_star.flatten()[:, None]
            x_data = X_star.flatten()[:, None]
            u_data = u.flatten()[:, None]

            t_data_f = t_data.copy()
            x_data_f = x_data.copy()

            if typ == 'train':
                idx = np.random.choice(np.where((x_data == -1) | (x_data == 1))[0], num_t)
                t_data = t_data[idx]
                x_data = x_data[idx]
                u_data = u_data[idx]

                init_idx = np.random.choice(N - 1, num_x - 4, replace=False) + 1
                t_data = np.concatenate([t_data, np.ones((2, 1)), np.zeros((num_x - 4, 1))], axis=0)
                x_data = np.concatenate([x_data, np.array([[-1], [1]]), x[init_idx]], axis=0)
                u_data = np.concatenate([u_data, -np.ones((2, 1)), u[init_idx, 0:1]], axis=0)

                return t_data, x_data, u_data, t_data_f, x_data_f

            else:
                return t_data_f, x_data_f

        def data_generator(self):
            '''
            t_data, x_data - выборка для сравнения с правильными данными (начальные и граничные условия)
            u_data = f(t_data, x_data)
            t_data_f, x_data_f - выборка для обучения физического аспекта модели
            '''
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            t_data, x_data, u_data, t_data_f, x_data_f = self.ac_generator(
                self.num_dots["train"],
                self.num_dots["physics"]
            )

            x_pairs = torch.FloatTensor(np.concatenate((t_data, x_data), axis=1)).to(device)
            x_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), axis=1)).to(device)
            x_f.requires_grad = True
            u_data = torch.FloatTensor(u_data).to(device)

            return {'main': x_f, 'secondary': x_pairs, 'secondary_true': u_data}

        def loss_calculator(self, u_pred_f, x_physics, u_pred, y_data):
            args = {'u_pred_f': u_pred_f, 'x_physics': x_physics}
            physics = self.equation(args)
            loss2 = torch.mean(physics ** 2)

            loss_u = torch.mean((u_pred - y_data) ** 2)  # use mean squared error
            loss = loss_u + loss2

            return loss

        def calculate_l2_error(self, path_true_data, model, device, test_data_generator):
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


    async def set_dataset(self):
        if not self.neural_model.data_set:
            # Создаем новый датасет
            dataset = self.mySpecialDataSet(
                num_dots={
                    "train": 201,
                    "physics": 513
                }
            )
            print('Creating new dataset')
            data = dataset.data_generator()
            # Сохраняем датасет в базу
            await dataset.insert()

            # Обновляем ссылку на датасет в модели
            self.neural_model.data_set = [dataset]
            await mNeuralNetMongo.m_save(self.neural_model)
        else:
            print('Loading existing dataset')
            # Получаем данные из существующего датасета
            existing_data = self.neural_model.data_set[0]

            # Создаем новый экземпляр mySpecialDataSet с теми же данными
            dataset = self.mySpecialDataSet(**existing_data.model_dump())

            # Заменяем старый датасет новым в модели
            self.neural_model.data_set = [dataset]
            data = dataset.data_generator()

        self.variables_f = data['main'].to(self.mydevice)
        self.u_data = data['secondary_true'].to(self.mydevice)
        self.variables = data['secondary'].to(self.mydevice)

    async def set_optimizer(self, opti = None):
        if opti is None:
            if self.neural_model.optimizer:
                opti = self.neural_model.optimizer[0]
                await opti.save()
            else:
                await self.abs_set_optimizer()
                opti = self.neural_model.optimizer[0]
                print('Создан новый оптимизатор:', opti)

        # Создаем PyTorch оптимизатор в зависимости от метода
        if opti.method == 'Adam':
            print('Используем Adam')
            self.torch_optimizer = torch.optim.Adam(
                self.mymodel.parameters(),
                **opti.params
            )
        elif opti.method == 'SGD':
            print('Используем SGD')
            self.torch_optimizer = torch.optim.SGD(
                self.mymodel.parameters(),
                **opti.params
            )
        elif opti.method == 'RMSprop':
            print('Используем RMSprop')
            self.torch_optimizer = torch.optim.RMSprop(
                self.mymodel.parameters(),
                **opti.params
            )
        elif opti.method == 'Adagrad':
            print('Используем Adagrad')
            self.torch_optimizer = torch.optim.Adagrad(
                self.mymodel.parameters(),
                **opti.params
            )
        else:
            raise ValueError(f"Неизвестный метод оптимизации: {opti.method}")

    async def construct_model(self, params, in_device):
        await self.create_model(params)

        self.mydevice = in_device
        # self.mymodel = pinn(params).to(self.mydevice)
        layers = [params.input_dim] + params.hidden_sizes + [params.output_dim]
        self.mymodel = pinn(layers, params.Fourier, params.FInputDim, params.FourierScale).to(self.mydevice)

        await self.set_optimizer()

    async def save_weights(self, path):
        print('run saving')
        weights = self.mymodel.state_dict()
        await self.abs_save_weights(weights)
        print('save complited')

    async def train(self):
        self.config = self.neural_model.hyper_param
        epochs = self.config.epochs
        for epoch in tqdm(range(epochs)):
            self.torch_optimizer.zero_grad()
            u_pred = self.mymodel(self.variables)
            u_pred_f = self.mymodel(self.variables_f)

            loss = self.neural_model.data_set[0].loss_calculator(u_pred_f, self.variables_f, u_pred, self.u_data)
            loss.backward()
            self.torch_optimizer.step()

            current_loss = loss.item()
            self.loss_history.append(current_loss)

            # if self.neural_model.data_set[0].calculate_l2_error:
            #     l2_error = self.neural_model.data_set[0].calculate_l2_error(
            #                 self.config.path_true_data,
            #                 self.mymodel,
            #                 self.mydevice,
            #                 test_data_generator)
            #     self.l2_history.append(l2_error)
            l2_error = 0

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch

            if (epoch % 400 == 0):
                print(
                    f"Epoch {epoch}, Train loss: {current_loss}, L2: {l2_error if self.neural_model.data_set[0].calculate_l2_error else 0}")

        await self.save_weights(sys.path[0] + self.config.save_weights_path)
        print(f"Оптимизатор: {self.torch_optimizer.__class__.__name__}")

    async def calc(self):
        test_data, [t, x], [N, T] = test_data_generator()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_variables = torch.FloatTensor(test_data).to(device)
        with torch.no_grad():
            u_pred = self.mymodel(test_variables)
        u_pred = u_pred.cpu().numpy().reshape(N, T)

        self.neural_model.hyper_param.path_true_data = "/equations/allen_cahn_eq/AC.mat"
        data = scipy.io.loadmat(sys.path[0] + self.neural_model.hyper_param.path_true_data)
        Exact = np.real(data['uu'])
        err = u_pred - Exact

        err = np.linalg.norm(err, 2) / np.linalg.norm(Exact, 2)
        print(f"L2 Relative Error: {err}")

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

        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

        await self.abs_save_plot(my_base64_jpgData)

        plt.clf()

        return my_base64_jpgData