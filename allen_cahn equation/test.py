import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat

# Параметры задачи
ε = 0.0001
λ = 5

# Загрузка данных
def get_dataset():
    data = loadmat("./allen_cahn equation/dataset/allen_cahn.mat")
    u_ref = data["usol"]  # настоящее решение
    t_star = data["t"].flatten()  # временные координаты
    x_star = data["x"].flatten()  # пространственные координаты
    return u_ref, t_star, x_star

# Определение архитектуры PINN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    
    def forward(self, t, x):
        inputs = torch.cat((t, x), dim=1)  # Объединяем t и x
        for i, layer in enumerate(self.layers[:-1]):
            inputs = torch.tanh(layer(inputs))  # tanh активация
        return self.layers[-1](inputs)  # Линейный выход

# Функция потерь
def loss_pinn(model, t, x, u_true=None):
    t.requires_grad_(True)
    x.requires_grad_(True)
    u_pred = model(t, x)
    
    # Вычисляем производные
    u_t = torch.autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    # Уравнение Аллена-Кана
    f = u_t - ε * u_xx + λ * u_pred**3 - λ * u_pred
    
    # Потери по данным
    loss_eq = torch.mean(f**2)
    loss_ic = torch.mean((u_pred[:t_ic.shape[0]] - u_ic)**2) if u_true is not None else 0
    loss_bc = torch.mean(u_pred[bc_mask]**2)
    
    return loss_eq + loss_ic + loss_bc

# Данные и обучение
u_ref, t_star, x_star = get_dataset()
t_star = torch.tensor(t_star, dtype=torch.float32).unsqueeze(1)
x_star = torch.tensor(x_star, dtype=torch.float32).unsqueeze(1)
u_ref = torch.tensor(u_ref, dtype=torch.float32)

# Подготовка данных
t_ic = t_star[:1]  # Начальные данные
x_ic = x_star
u_ic = u_ref[0, :]  # Начальное значение u(t=0, x)

# Граничные условия
bc_mask = (x_star == -1) | (x_star == 1)

# Инициализация PINN
layers = [2, 64, 64, 64, 1]  # 2 входа (t, x) -> несколько скрытых слоев -> 1 выход (u)
model = PINN(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Обучение
for epoch in range(10000):
    optimizer.zero_grad()
    loss = loss_pinn(model, t_star, x_star, u_ref)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Предсказание и визуализация
