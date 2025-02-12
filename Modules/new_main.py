import ml_collections
import torch
from Modules.pinn_init_torch import pinn
from Modules.optim_Adam_torch import create_optim
from Modules.train_torch import Train_torch

class Trainable():
    
    equation: str
    
    def __init__(self, 
                 name:str):
        self.equation = name
        
        cfg = ml_collections.ConfigDict()

        # initial parameters for optimizer
        optimizers = ["Adam", "NAdam", "LBFGS"]
        cfg.optimizer = optimizers[0]
        cfg.lr = 1e-4
        cfg.betas=(0.9, 0.999)

        #initial parameters for nn
        cfg.hidden_count = 32
        cfg.input_dim = 1
        cfg.output_dim = 1
        cfg.hidden_sizes = [32, 32, 32]

        #Fourier
        cfg.Fourier = False
        cfg.FinputDim = None
        cfg.FourierScale = None

        #Test data
        cfg.num_dots = [400]

        #Train
        cfg.epochs = 10000
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg.num_dots = [400, 50]
        
        #путь файла с данными относительно директории проекта
        cfg.path_true_data = "/data/OSC.npy"
        cfg.save_weights_path = "/osc_1d.pth"
        
        #initialization model and optimizer
        self.model = pinn(cfg)
        self.optimizer = create_optim(self.model, cfg) 
        
        self.trainer
        
        if (self.equation == "oscillator"):
            from Modules.oscillator.data_generator import data_generator
            from Modules.oscillator.loss_calc import loss_calculator
            from Modules.oscillator.calculate_l2 import calculate_l2_error
            from Modules.oscillator.test_data_generator import generator as test_data_generator
            from Modules.oscillator.vizualizer import vizualize        
            self.trainer = Train_torch(cfg,
                        self.model, 
                        self.optimizer, 
                        data_generator, 
                        loss_calculator,
                        test_data_generator,
                        calculate_l2_error,
                        vizualize)
    
    def train(self):
        self.trainer.train
    
    def inference(self):
        self.trainer.printEval
        
        