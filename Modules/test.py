import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Гиперпараметры
epochs = 100000
lr = 1e-3
epsilon = 0.01
hidden_units = 32

# Нейросеть
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.fc(inputs)

# Генерация данных
def generate_data(n=1000):
    # Коллокационные точки (x, t)
    x = torch.rand(n, 1) * 2 - 1  # [-1, 1]
    t = torch.rand(n, 1)          # [0, 1]
    
    # Начальные условия (t=0)
    x_ic = torch.rand(n//10, 1)*2 - 1
    t_ic = torch.zeros(n//10, 1)
    u_ic = x_ic**2 * torch.cos(np.pi * x_ic)  # Пример начального условия
    
    return (x, t), (x_ic, t_ic, u_ic)

# Физические законы (уравнение Аллена-Кана)
def physics_loss(model, x, t, epsilon):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # Вычисление производных
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                             create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                             create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                              create_graph=True)[0]
    
    # Остаток уравнения
    residual = u_t - epsilon**2 * u_xx - u + u**3
    return torch.mean(residual**2)

# Обучение
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Генерация данных
(col_x, col_t), (x_ic, t_ic, u_ic) = generate_data()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Потери на уравнении
    loss_pde = physics_loss(model, col_x, col_t, epsilon)
    
    # Потери на начальных условиях
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    
    # Общий loss
    loss = loss_pde + loss_ic
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.20f}")

# Визуализация
def plot_solution(model):
    x = torch.linspace(-1, 1, 100).view(-1, 1)
    t = torch.linspace(0, 1, 100).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze())
    
    with torch.no_grad():
        U = model(X.reshape(-1, 1), T.reshape(-1, 1))
        U = U.view(100, 100).numpy()
    
    plt.pcolormesh(X, T, U, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()
    plt.show()

plot_solution(model)