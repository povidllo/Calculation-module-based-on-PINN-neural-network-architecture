'''


    metric_data - это тип вычисления лосс функции тип metric_data = nn.MSELoss() (хз правильно ли сказал, но думаю понятно)
    device - это девайс
    model - модель
    устанавливает оптимизатор

    сделай функцию тип init_nn(типо интерфейса), где она инициализируется и устанавливает значения выше
'''

import torch
import torch.nn as nn

class init_nn(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        n_units=100,
        epochs=1000,
        loss = nn.MSELoss(), # лосс функция для экспериментальных данных
        loss1=None, # лосс функция для граничных условий
        lr=1e-3,
        loss2=None, # лосс функция для дифф. уравнения
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
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self) + self.loss1(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()