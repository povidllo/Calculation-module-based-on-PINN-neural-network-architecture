import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import io
import base64
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
from torch.autograd import Variable

from optim_Adam_torch import create_optim
from pinn_init_torch import pinn
from equations.oscillator_eq.test_data_generator import generator as test_data_generator
from mongo_schemas import *
from mNeural_abs import *


# torch.manual_seed(123)
# np.random.seed(44)
# torch.cuda.manual_seed(44)

class oscillator_nn(AbsNeuralNet):
    mymodel = None
    mydevice = None
    torch_optimizer = None

    loss_history = []
    l2_history = []
    best_loss = float('inf')
    best_epoch = 0

    class mySpecialDataSet(AbsNeuralNet.AbsDataSet):
        def oscillator(self, x, d=2, w0=20):
            '''
            Решение уравнения, которое должна получить нейросеть
            '''
            assert d < w0
            w = np.sqrt(w0**2-d**2)
            phi = np.arctan(-d/w)
            A = 1/(2*np.cos(phi))
            cos = torch.cos(phi+w*x)
            sin = torch.sin(phi+w*x)
            exp = torch.exp(-d*x)
            y = exp*2*A*cos
            return y

        def time_generator(self, num_t, num_ph, typ='train'):
            '''
            Генерирует данные (временные координаты) для обучения осциллятора
            t - вся выборка
            t_phys - выборка для обучения физического аспекта модели
            t_data - выборка для обучения путем сравнения с правильными данными
            '''
            t = np.linspace(0, 1, num_t).reshape(-1, 1)
            l = len(t)//2
            t_data = t[0:l:l//10]

            t_phys = np.linspace(0, 1, num_ph).reshape(-1, 1)
            return t, t_data, t_phys

        def data_generator(self):
            '''
            x - Вся выборка
            x_data - Выборка для обучения путем сравнения с правильными данными
            x_physics - Выборка для обучения физического аспекта модели
            '''
            x, x_data, x_physics = self.time_generator(
                self.num_dots["train"],
                self.num_dots["physics"]
            )
            x = torch.FloatTensor(x)
            x_data = torch.FloatTensor(x_data)
            x_physics = torch.FloatTensor(x_physics).requires_grad_(True)
            y = self.oscillator(x).view(-1,1)
            # y_data = y[0:len(x)//2:len(x)//20]
            y_data = self.oscillator(x_data).view(-1,1)
            # Сохраняем данные в base64
            buffer = io.BytesIO()
            np.savez_compressed(buffer,
                x=x.cpu().detach().numpy(),
                x_data=x_data.cpu().detach().numpy(),
                x_physics=x_physics.cpu().detach().numpy(),
                y=y.cpu().detach().numpy(),
                y_data=y_data.cpu().detach().numpy()
            )
            encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Сохраняем в базу данных
            self.params['points_data'] = encoded_data

            # Для обратной совместимости пока оставляем и старое сохранение
            # np.save(sys.path[0] + '/data/OSC.npy', y.cpu().detach().numpy())

            return {'main': x_physics, 'secondary': x_data, 'secondary_true': y_data}

        def equation(self, args):
            '''
            Уравнение затухающего гармонического осциллятора
            '''
            yhp, x_physics, d, w0 = args.values()
            mu = d * 2
            k = w0 ** 2
            dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
            dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]
            physics = dx2 + mu*dx + k*yhp
            return physics

        def loss_calculator(self, u_pred_f, x_physics, u_pred, y_data):
            loss1 = torch.mean((u_pred-y_data)**2)# use mean squared error

            args = {'yhp':u_pred_f, 'x_physics': x_physics, 'd':2, 'w0':20}
            physics = self.equation(args)
            loss2 = (1e-4)*torch.mean(physics**2)

            loss = loss1 + loss2
            return loss

        def calculate_l2_error(self, path_true_data, model, device, test_data_generator):
            x, _, _ = test_data_generator()

            # Загружаем данные из базы вместо файла
            if 'points_data' not in self.params:
                print("Warning: No data found in database, using file")
                y = np.load(sys.path[0] + path_true_data)
            else:
                # Декодируем данные из base64
                decoded_data = base64.b64decode(self.params['points_data'])
                buffer = io.BytesIO(decoded_data)
                data = np.load(buffer)
                y = data['y']

            u_pred = model(x)
            u_pred = u_pred.cpu().detach().numpy()
            true = y
            # Сравнение с эталоном
            error = np.linalg.norm(u_pred - true, 2) / np.linalg.norm(true, 2)
            return error

        # Генерация равномерно распределённых точек по заданным диапазонам
        def generate_linear_points(self, num_points, ranges):
            grids = [torch.linspace(r[0], r[1], num_points).to(self.device) for r in
                     ranges]  # Создание равномерных точек в каждом диапазоне
            return torch.stack(torch.meshgrid(*grids, indexing='ij')).reshape(len(ranges),
                                                                              -1)  # Объединение в сетку и преобразование в двумерный тензор

        # Генерация случайных точек по заданным диапазонам
        def generate_random_points(self, num_points, ranges):
            # Генерация случайных значений по каждому диапазону, общее число точек — num_points ** dim
            grids = [torch.Tensor(num_points ** (len(ranges))).to(self.device).uniform_(r[0], r[1]) for r in ranges]
            return torch.stack(grids).reshape(len(ranges), -1)  # Объединение в матрицу размером (dim, N)

        # Вычисляет количество линейных точек как линейную функцию от N
        def l(self, N):
            return int(self.linear_mult * N)

        # Основной метод генерации распределённых точек для обучения PINN
        def make_distributed_points(self):
            x_range = [0, 1]
            y_range = [0, 1]
            t_range = [0, self.T * self.t_max]

            # --- Начальные условия ---
            with torch.no_grad():  # Без вычисления градиентов
                x_linear, y_linear = self.generate_linear_points(self.l(self.N_IC), [x_range, y_range])
                x_random, y_random = self.generate_random_points(self.N_IC, [x_range, y_range])

            # Объединяем линейные и случайные точки начального условия
            self.x_IC = Variable(torch.cat((x_linear, x_random)), requires_grad=True).to(self.device)
            self.y_IC = Variable(torch.cat((y_linear, y_random)), requires_grad=True).to(self.device)
            self.t_IC = Variable(torch.zeros_like(self.x_IC), requires_grad=True).to(self.device)  # t = 0 для IC
            if self.NN_params['input_size'] != 3:
                self.beta_IC = Variable(self.beta * torch.ones_like(self.x_IC), requires_grad=True).to(self.device)

            # Вызываем пользовательскую функцию для инициализации значений в начальных точках
            self.Initial_conditions(self.x_IC, self.y_IC)

            # --- Точки для уравнения (PDE Points) ---
            # Линейные и случайные точки в пространстве и времени
            x_linear, y_linear, t_linear = self.generate_linear_points(self.l(self.N_PDE), [x_range, y_range, t_range])
            x_random, y_random, t_random = self.generate_random_points(self.N_PDE, [x_range, y_range, t_range])

            # Инициализация, если PDE точки ещё не заданы
            try:
                self.x_PDE
            except AttributeError:
                self.x_PDE = torch.Tensor([]).to(self.device)
                self.y_PDE = torch.Tensor([]).to(self.device)
                self.t_PDE = torch.Tensor([]).to(self.device)

            # Объединяем новые случайные точки с ранее сгенерированными (отбрасывая старые)
            x = Variable(torch.cat((x_random, self.x_PDE[self.N_PDE ** 3:])), requires_grad=True).to(self.device)
            y = Variable(torch.cat((y_random, self.y_PDE[self.N_PDE ** 3:])), requires_grad=True).to(self.device)
            t = Variable(torch.cat((t_random, self.t_PDE[self.N_PDE ** 3:])), requires_grad=True).to(self.device)

            # Вычисление уравнений (конвекция, дивергенция, корреляция)
            if self.NN_params['input_size'] == 3:
                conv, div, corr = self.compute_PDE(x, y, t)
            else:
                beta = Variable(self.beta * torch.ones_like(x), requires_grad=True).to(self.device)
                conv, div, corr = self.compute_PDE(x, y, t, beta)

            # Адаптивное распределение на основе ошибки (если ошибка > 0.01, то включаем точку)
            pde_dist = self.weights[0] * torch.where(conv.abs() > 0.01, 1, 0) + \
                       self.weights[1] * torch.where(div.abs() > 0.01, 1, 0) + \
                       self.weights[2] * torch.where(corr.abs() > 0.01, 1, 0)

            # Семплируем точки по распределению ошибки
            sampled_indices_pde = torch.multinomial(pde_dist / pde_dist.sum(), self.N_PDE ** 3, replacement=True)

            # Обновляем точки PDE (добавляем линейные + адаптивно выбранные)
            self.x_PDE = Variable(torch.cat((x_linear, x[sampled_indices_pde])), requires_grad=True)
            self.y_PDE = Variable(torch.cat((y_linear, y[sampled_indices_pde])), requires_grad=True)
            self.t_PDE = Variable(torch.cat((t_linear, t[sampled_indices_pde])), requires_grad=True)
            if self.NN_params['input_size'] != 3:
                self.beta_PDE = Variable(self.beta * torch.ones_like(self.x_PDE), requires_grad=True).to(self.device)

            # --- Граничные условия ---
            with torch.no_grad():
                # Линейные точки на границах
                x = torch.linspace(0, 1, self.N_BC).to(self.device)
                y = torch.linspace(0, 1, self.N_BC).to(self.device)
                t = torch.linspace(0, self.T * self.t_max, self.N_BC).to(self.device)
                # c_condition_linear = cnd.form_boundaries([x, y, t], self.ones, self.zeros)
                c_condition_linear = torch.linspace(0, 1, self.N_BC).to(self.device)

                # Случайные точки на границах
                x = torch.Tensor(self.N_BC).to(self.device).uniform_(0, 1)
                y = torch.Tensor(self.N_BC).to(self.device).uniform_(0, 1)
                t = torch.Tensor(self.N_BC).to(self.device).uniform_(0, self.T * self.t_max)
                # c_condition_random = cnd.form_boundaries([x, y, t], self.ones, self.zeros)
                c_condition_random = torch.linspace(0, 1, self.N_BC).to(self.device)

                # Объединяем точки и делаем их дифференцируемыми
                self.x_BC = Variable(torch.cat((c_condition_linear[:, 0], c_condition_random[:, 0])),
                                     requires_grad=True)
                self.y_BC = Variable(torch.cat((c_condition_linear[:, 1], c_condition_random[:, 1])),
                                     requires_grad=True)
                self.t_BC = Variable(torch.cat((c_condition_linear[:, 2], c_condition_random[:, 2])),
                                     requires_grad=True)

                # Вычисление граничных условий
                if self.NN_params['input_size'] == 3:
                    self.c, self.p, self.u = self.Boundary_conditions(self.x_BC, self.y_BC, self.t_BC, self.beta)
                else:
                    self.beta_BC = Variable(self.beta * torch.ones_like(self.x_BC), requires_grad=True)
                    self.c, self.p, self.u = self.Boundary_conditions(self.x_BC, self.y_BC, self.t_BC, self.beta_BC)

                # Разделение на разные участки границы
                self.where_c_tb = (self.y_BC == 1) | (self.y_BC == 0)  # верх/низ
                self.where_c_in = (self.x_BC == 0) & (
                            (self.y_BC - 1 / 2).abs().round(decimals=5) <= self.zeta / 2)  # вход
                self.where_c_out = (self.x_BC == 0) & (
                            (self.y_BC - 1 / 2).abs().round(decimals=5) > self.zeta / 2)  # выход

                # Обработка граничных точек для разных участков
                self.x_lr = Variable(self.x_BC[self.where_c_out], requires_grad=True).to(self.device)
                self.y_lr = Variable(self.y_BC[self.where_c_out], requires_grad=True).to(self.device)
                self.t_lr = Variable(self.t_BC[self.where_c_out], requires_grad=True).to(self.device)

                self.x_tb = Variable(self.x_BC[self.where_c_tb], requires_grad=True).to(self.device)
                self.y_tb = Variable(self.y_BC[self.where_c_tb], requires_grad=True).to(self.device)
                self.t_tb = Variable(self.t_BC[self.where_c_tb], requires_grad=True).to(self.device)

                if self.NN_params['input_size'] != 3:
                    self.beta_lr = Variable(self.beta * torch.ones_like(self.x_lr), requires_grad=True)
                    self.beta_tb = Variable(self.beta * torch.ones_like(self.x_tb), requires_grad=True)


    async def set_dataset(self):
        if not self.neural_model.data_set:
            # Создаем новый датасет
            dataset = self.mySpecialDataSet(
                params={'nu': 3},
                num_dots={
                    "train": 400,
                    "physics": 50
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
        start_time = time.time()
        for epoch in tqdm(range(epochs)):
            self.torch_optimizer.zero_grad()
            u_pred = self.mymodel(self.variables)
            u_pred_f = self.mymodel(self.variables_f)

            loss = self.neural_model.data_set[0].loss_calculator(u_pred_f, self.variables_f, u_pred, self.u_data)
            loss.backward()
            self.torch_optimizer.step()

            current_loss = loss.item()
            self.loss_history.append(current_loss)

            if self.neural_model.data_set[0].calculate_l2_error:
                l2_error = self.neural_model.data_set[0].calculate_l2_error(
                            self.config.path_true_data,
                            self.mymodel,
                            self.mydevice,
                            test_data_generator)
                self.l2_history.append(l2_error)

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch

            if(epoch % 400 == 0 and epoch != epochs-1):
                self.add_to_loss_graph(time.time() - start_time, current_loss, epoch)
                print(f"Epoch {epoch}, Train loss: {current_loss}, L2: {l2_error if self.neural_model.data_set[0].calculate_l2_error else 0}")
            if(epoch == epochs - 1):
                self.add_to_loss_graph(time.time() - start_time, current_loss, epoch)

        await self.save_weights(sys.path[0] + self.config.save_weights_path)
        await self.set_loss_graph()
        print("loss data " + str(self.loss_graph))
        print(f"Оптимизатор: {self.torch_optimizer.__class__.__name__}")

    async def calc(self):
        x, _, _ = test_data_generator()

        if 'points_data' not in self.neural_model.data_set[0].params:
            print("Warning: No data found in database, using file")
            y = np.load(sys.path[0] + self.neural_model.hyper_param.path_true_data)
        else:
            decoded_data = base64.b64decode(self.neural_model.data_set[0].params['points_data'])
            buffer = io.BytesIO(decoded_data)
            data = np.load(buffer)
            y = data['y']

        u_pred = self.mymodel(x)
        u_pred = u_pred.cpu().detach().numpy()
        true = y
        err = u_pred - true

        err = np.linalg.norm(err, 2) / np.linalg.norm(true, 2)
        print(f"L2 Relative Error: {err}")

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

        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

        await self.abs_save_plot(my_base64_jpgData)

        plt.clf()

        return my_base64_jpgData
