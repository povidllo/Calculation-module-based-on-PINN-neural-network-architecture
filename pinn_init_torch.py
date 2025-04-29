# """
#     Этот файл занимается инициализацией базовой нейронной сети PINN
# """


# import torch
# from torch import nn

# """
#     Этот класс отвечает за Fourier feature для исходной нейронной сети
# """
# class FourierEmbs(nn.Module):
#     def __init__(self, input_dim, embed_dim, embed_scale):
#         """
#         Реализация слоя Фурье-эмбеддингов.
#         :param input_dim: Размер входных данных.
#         :param embed_dim: Размер выходных эмбеддингов (должен быть чётным).
#         :param embed_scale: Масштаб для инициализации случайных частот.
#         """
#         super().__init__()
#         self.kernel = nn.Parameter(
#             torch.randn(input_dim, embed_dim // 2) * embed_scale, requires_grad=False
#         )  # Фиксированные частоты

#     def forward(self, x):
#         projection = torch.matmul(x, self.kernel)
#         return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)

# """
#     Этот класс создает исходную нейронную сеть PINN
#     Он принимает параметры:
#         input_dim,   Размер входного слоя
#         output_dim,   Размер выходного слоя
#         hidden_sizes,   Массив размеров внутренних слоев
#         fourier=False,   Булевое значение, отвечающее за необходимость испгользования Fourier feature
#         fourier_dim=None,   Размерность Fourier feature
#         fourier_scale=1.0,   Множитель для Fourier feature

# """
# class PINN(nn.Module):  
#     def __init__(self, 
#                  input_dim,  
#                  output_dim,
#                  hidden_sizes, 
#                  fourier=False,
#                  fourier_dim=None,
#                  fourier_scale=1.0):
#         super().__init__()
#         self.use_fourier = fourier
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         if self.use_fourier:
#             self.fourier_layer = FourierEmbs(
#                 input_dim=input_dim, 
#                 embed_dim=fourier_dim, 
#                 embed_scale=fourier_scale
#             ).to(device)
#             input_dim = fourier_dim  # После Фурье-преобразования входная размерность увеличивается
        
#         # Создаем список слоев
#         layers = []
#         prev_size = input_dim
        
#         # Добавляем внутренние слои
#         for hidden_size in hidden_sizes:
#             layers.append(nn.utils.weight_norm(nn.Linear(prev_size, hidden_size), dim=0))
#             layers.append(nn.Tanh())
#             prev_size = hidden_size
        
#         # Добавляем выходной слой
#         layers.append(nn.Linear(prev_size, output_dim))
        
#         # Собираем все слои в последовательность
#         self.layers_stack = nn.Sequential(*layers)
    
#     def forward(self, x):
#         if self.use_fourier:
#             x = self.fourier_layer(x)  
#         return self.layers_stack(x)

# """ 
#     Инициализация весов с помощью Ксавье
# """
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_normal_(m.weight)

# """
#     Создаёт модель PINN с конфигурацией из `cfg`.
# """
# def pinn(cfg):
#     pytorch_model = PINN(
#         input_dim = cfg.input_dim,  
#         output_dim = cfg.output_dim,
#         hidden_sizes = cfg.hidden_sizes, 
#         fourier=cfg.Fourier,
#         fourier_dim=cfg.FInputDim if cfg.Fourier else None,
#         fourier_scale=cfg.FourierScale if cfg.Fourier else 1.0
#     )
#     pytorch_model.apply(weights_init)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     pytorch_model.to(device)
#     return pytorch_model

"""
    Этот файл занимается инициализацией базовой нейронной сети PINN
"""


import torch
from torch import nn



class LinearBlock(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(LinearBlock, self).__init__()
        self.layer = nn.utils.weight_norm(nn.Linear(in_nodes, out_nodes), dim=0)

    def forward(self, x):
        x = self.layer(x)
        x = torch.tanh(x)
        return x


class PINN(nn.Module):

    def __init__(self, layer_list):
        super(PINN, self).__init__()


        self.input_layer = nn.utils.weight_norm(nn.Linear(layer_list[0], layer_list[1]), dim=0)
        self.hidden_layers = self._make_layer(layer_list[1:-1])
        self.output_layer = nn.Linear(layer_list[-2], layer_list[-1])

    def _make_layer(self, layer_list):
        layers = []
        for i in range(len(layer_list) - 1):
            block = LinearBlock(layer_list[i], layer_list[i + 1])
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.tanh(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def pinn(layer_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('torch.cuda.is_available()', torch.cuda.is_available())
    model = PINN(layer_list)
    model.apply(weights_init)
    model.to(device)
    print(model)
    return model