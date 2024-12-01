import neural_network as net
import matplotlib.pyplot as plt
from neural_network import torch, nn
from neural_network import device
import numpy as np
import scipy.io

'''
du_dt - ε * d2u_d2x + λ * u(t,x)^3 - λ * u = 0

u(0,x) = x^2 * cos(πx)
u(t,-1) = u(t,1)
du_dx(t,-1) = du_dx(t,1)

ε = 0.0001
λ = 5
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка данных
def get_dataset():
    data = scipy.io.loadmat("./allen_cahn equation/dataset/allen_cahn.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    return u_ref, t_star, x_star

def print_results():
    u_ref, t_star, x_star = get_dataset()
    T, X = np.meshgrid(t_star, x_star)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, u_ref.T, cmap='jet')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_zlabel('u(t, x)')
    ax.set_title('Allen-Cahn Equation Solution')
    
    plt.tight_layout()
    plt.show()

def lossPDE(input, epsilon=0.0001, lamb=5):
    t = input[0]
    x = input[1]
    u = model(input).to(device)
    du_dt = torch.autograd.grad(u, t, torch.ones_like(t), create_graph=True, \
                                retain_graph=True)[0]
    dx_dx = torch.autograd.grad(u, x, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    d2xdt2 = torch.autograd.grad(dx_dx, t, torch.ones_like(t), create_graph=True, \
                            retain_graph=True)[0]
    # du_dt - ε * d2u_d2x + λ * u(x,t)^3 - λ * u = 0
    f1 = du_dt - epsilon*d2xdt2 + lamb*u**3 - lamb * u
    loss_pde = model.loss(f1, torch.zeros_like(f1))
    
    return loss_pde

def lossIC(input, true_out):
    inlet_mask = (t_tensor == 0)  # Маска для t == 0
    input_t0 = input_data[inlet_mask.repeat(len(x_tensor), 1).view(-1)]  # Применяем маску
    out_IC = out_dataset[0, :]
    # print(inlet_mask)
    # print("input data с маской на 0:", input_t0)
    out_nn = model(input_t0)
    loss_ic = model.loss(out_IC - out_nn)
    return loss_ic
    

out_dataset, t, x = get_dataset()
# print(len(out_dataset[0]))
# print(len(x))
# print(len(t))

x_tensor = torch.tensor(x, dtype=torch.float32)

# print("x: ", x_tensor.shape, " ", x_tensor)
# print("\n\n\n---------------")
# print("t: ", t_tensor.shape, " ", t_tensor)

x_all_t = x_tensor.repeat(len(t_tensor), 1).T
t_all_x = t_tensor.repeat(len(x_tensor), 1)
# print("Shape of x_all_t:", x_all_t.shape)
# print("Shape of t_all_x:", t_all_x.shape)

# prepared_data = torch.stack((x_all_t, t_all_x), dim=2) #(x, t)
prepared_data = torch.stack((t_all_x, x_all_t), dim=2)  #(t, x)
# print(prepared_data.shape)



# Преобразуем в одномерный массив для подачи в модель
# Каждый элемент будет иметь форму (x, t) для подачи в модель
# Сначала делаем данные одномерными
input_data = prepared_data.view(-1, 2)  # Преобразуем в форму (len(x_star) * len(t_star), 2)

# Преобразование в нужный формат для модели (например, [batch_size, input_size])
# print("input data:", input_data)


print_results()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# global model 
# model = net.simpleModel('''!!!!!!!!''').to(device)
