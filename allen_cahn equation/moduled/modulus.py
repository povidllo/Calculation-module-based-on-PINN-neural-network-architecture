import torch
from torch import nn


class FourierEmbs(nn.Module):
    def __init__(self, input_dim, embed_dim, embed_scale):
        """
        Реализация слоя Фурье-эмбеддингов.
        :param input_dim: Размер входных данных.
        :param embed_dim: Размер выходных эмбеддингов (должен быть чётным).
        :param embed_scale: Масштаб для инициализации случайных частот.
        """
        super().__init__()
        self.kernel = nn.Parameter(
            torch.randn(input_dim, embed_dim // 2) * embed_scale, requires_grad=False
        )  # Фиксированные частоты

    def forward(self, x):
        projection = torch.matmul(x, self.kernel)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class PINN(nn.Module):
    def __init__(self, 
                 hidden_size,
                 fourier=False,
                 fourier_dim=None,
                 fourier_scale=1.0):
        super().__init__()
        self.use_fourier = fourier
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        input_dim = 2  # Размерность входных данных
        if self.use_fourier:
            self.fourier_layer = FourierEmbs(
                input_dim=input_dim, 
                embed_dim=fourier_dim, 
                embed_scale=fourier_scale
            ).to(device)
            input_dim = fourier_dim  # После Фурье-преобразования входная размерность увеличивается
        
        # Последовательность полносвязных слоёв
        self.layers_stack = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim, hidden_size), dim=0),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size), dim=0),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size), dim=0),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size), dim=0),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, x):
        if self.use_fourier:
            x = self.fourier_layer(x)  
        return self.layers_stack(x)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def pinn(cfg):
    """
    Создаёт модель PINN с конфигурацией из `cfg`.
    """
    model = PINN(
        hidden_size=cfg.hidden_count,
        fourier=cfg.Fourier,
        fourier_dim=cfg.FinputDim if cfg.Fourier else None,
        fourier_scale=cfg.FourierScale if cfg.Fourier else 1.0
    )
    model.apply(weights_init)
    return model
