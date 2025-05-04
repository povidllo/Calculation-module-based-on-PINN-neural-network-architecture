import torch
from torch import nn
import math



class LinearBlock(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(LinearBlock, self).__init__()
        self.layer = nn.utils.weight_norm(nn.Linear(in_nodes, out_nodes), dim=0)

    def forward(self, x):
        x = self.layer(x)
        x = torch.tanh(x)
        return x


class FourierFeature(nn.Module):
    def __init__(self, in_features, mapping_size=256, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn((in_features, mapping_size)) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PINN(nn.Module):

    def __init__(self, layer_list, use_fourier=False, fourier_dim=256, fourier_scale=10.0):
        super(PINN, self).__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeature(layer_list[0], mapping_size=fourier_dim, scale=fourier_scale)
            input_dim = fourier_dim * 2
        else:
            input_dim = layer_list[0]
        self.input_layer = nn.utils.weight_norm(nn.Linear(input_dim, layer_list[1]), dim=0)
        self.hidden_layers = self._make_layer(layer_list[1:-1])
        self.output_layer = nn.Linear(layer_list[-2], layer_list[-1])

    def _make_layer(self, layer_list):
        layers = []
        for i in range(len(layer_list) - 1):
            block = LinearBlock(layer_list[i], layer_list[i + 1])
            print(block)
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)
        x = self.input_layer(x)
        x = torch.tanh(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def pinn(layer_list, use_fourier=False, fourier_dim=256, fourier_scale=10.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('torch.cuda.is_available()', torch.cuda.is_available())
    model = PINN(layer_list, use_fourier=use_fourier, fourier_dim=fourier_dim, fourier_scale=fourier_scale)
    model.apply(weights_init)
    model.to(device)
    return model