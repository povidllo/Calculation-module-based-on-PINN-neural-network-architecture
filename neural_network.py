"""


    metric_data - это тип вычисления лосс функции тип metric_data = nn.MSELoss() (хз правильно ли сказал, но думаю понятно)
    device - это девайс
    model - модель
    устанавливает оптимизатор

    сделай функцию тип init_nn(типо интерфейса), где она инициализируется и устанавливает значения выше
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)

class init_nn(nn.Module):
    def __init__(
            self,
            input_dim=1,
            output_dim=1,
            n_units=100,
            epochs=1000,
            loss=nn.MSELoss(),  # лосс функция для экспериментальных данных
            loss1=None,  # лосс функция для граничных условий
            lr=0.1,
            loss2=None,  # лосс функция для дифф. уравнения
            loss2_weight=1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X):
        # Xt = np_to_th(X)
        pbar = tqdm(range(self.epochs), desc='Training Progress')
        writer = SummaryWriter()
        optimiser = optim.LBFGS(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for step in range(self.epochs):
            # optimiser.zero_grad()
            # outputs = self.forward(X)
            # loss = self.loss2_weight * (self.loss2(X) + self.loss1(X))
            # loss.backward()
            def closure():
                optimiser.zero_grad()
                loss = self.loss2_weight * (self.loss2(X) + 1e3*self.loss1(X))
                loss.backward()
                return loss
            optimiser.step(closure)
            loss = closure()
            if step % 2 == 0:
                current_loss = closure().item()
                pbar.set_description("Step: %d | Loss: %.6f" %
                                    (step, current_loss))
                writer.add_scalar('Loss/train', current_loss, step)
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()
