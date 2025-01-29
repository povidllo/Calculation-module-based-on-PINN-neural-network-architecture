from tqdm import tqdm
import torch

def ac_equation(u, tx):
    u_tx = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph= True)[0]
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph= True)[0][:, 1:2]
    e = u_t -0.0001*u_xx + 5*u**3 - 5*u
    return e

class Train_torch:
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 data_generator,
                 loss_calculator,
                 vizualizer = None,
                 error_metric = None,
                 calculate_l2_error=None):
        
        # Привязка основной
        self.config = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = cfg.epochs
        
        
        
        # Привязка внешних модулей
        self.data_generator = data_generator
        self.loss_calculator = loss_calculator
        self.vizualizer = vizualizer
        self.error_metric = error_metric
        self.calculate_l2_error = calculate_l2_error
        # Подготовка данных
        self.__prepare_data()
        
        # Логирование
        self.loss_history = []
        self.l2_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0    

    def __prepare_data(self):
        """Загрузка и подготовка данных через внешний модуль"""
        data = self.data_generator(self.config)
        self.variables_f = data['train'].to(self.device)
        self.u_data = data['boundary_true'].to(self.device)
        self.variables = data['boundary'].to(self.device)

    def save_model(self, path):
        """Сохранение модели"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Загрузка модели"""
        self.model.load_state_dict(torch.load(path))


    def train(self):
        """Обучение модели"""
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.optimizer.zero_grad()
            
            # Предсказания
            u_pred = self.model(self.variables)
            u_pred_f = self.model(self.variables_f)
            
            # Вычисление функции потерь
            loss = self.loss_calculator(u_pred_f, self.variables_f, u_pred, self.u_data)
            
            # Обновление весов
            loss.backward()
            self.optimizer.step()
            
            # Логирование
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            if self.calculate_l2_error:
                l2_error = self.calculate_l2_error()
                self.l2_history.append(l2_error)            
            
            # Сохранение лучшей модели
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                self.save_model(f'./ac_1d.pth')
                
            
            # Логирование
            if epoch:
                print(f"Epoch {epoch}, Train loss: {current_loss}, L2: {123}")

