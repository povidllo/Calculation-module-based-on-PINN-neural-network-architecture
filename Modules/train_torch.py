from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

class Train_torch:
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 data_generator,
                 loss_calculator,
                 test_data_generator=None,
                 calculate_l2_error=None,
                 vizualizer = None):
        
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
        self.calculate_l2_error = calculate_l2_error
        self.test_data_generator = test_data_generator
        # Подготовка данных
        self.__prepare_data()
        
        # Логирование
        self.loss_history = []
        self.l2_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0    

    def __prepare_data(self):
        '''
        Загрузка и подготовка данных через внешний модуль
        main - данные для обучения физического аспекта модели
        secondary - данные для обучения, путем сравнения с правильными данными
        secondary_true - правильные данные для secondary
        '''
        data = self.data_generator(self.config)
        self.variables_f = data['main'].to(self.device)
        self.u_data = data['secondary_true'].to(self.device)
        self.variables = data['secondary'].to(self.device)
    
    def printLossGraph(self):
        '''
        Выводит график функции потерь, а также эпоху с наименьшей величиной потерь
        '''
        print(f"[Best][Epoch: {self.best_epoch}] Train loss: {self.best_loss}")
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.l2_history)
        plt.show()

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
                l2_error = self.calculate_l2_error(self.config.path_true_data, self.model, self.device, self.test_data_generator)
                self.l2_history.append(l2_error)            
            
            # Сохранение лучшей модели
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                
                self.save_model(sys.path[0] + self.config.save_weights_path)
                
            
            # Логирование
            if(epoch % 400 == 0):
                print(f"Epoch {epoch}, Train loss: {current_loss}, L2: {l2_error if self.calculate_l2_error else 0}")

    def printEval(self):
        #Загружаем лучшие веса
        self.model.load_state_dict(torch.load(sys.path[0] + self.config.save_weights_path))
        self.vizualizer(self.config.path_true_data, self.model, self.device, self.test_data_generator)