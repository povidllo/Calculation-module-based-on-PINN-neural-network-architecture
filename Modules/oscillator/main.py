import torch
import sys
import os
from pprint import pprint
import numpy as np

# Добавляем родительскую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Modules.pinn_init_torch import pinn
from Modules.optim_Adam_torch import create_optim
from Modules.train_torch import Train_torch
from Modules.oscillator.data_generator import data_generator
from Modules.oscillator.loss_calc import loss_calculator
from Modules.oscillator.calculate_l2 import calculate_l2_error
from Modules.oscillator.test_data_generator import generator as test_data_generator
from Modules.oscillator.vizualizer import vizualize
# import cfg_pinn_init as cfg_pinn_init
import cfg_main as cfg_main
# import cfg_train_torch as cfg_train_torch

torch.manual_seed(123)
# np.random.seed(44)
# torch.cuda.manual_seed(44)

model = pinn(cfg_main.get_config())
# Вывод весов при инициализации
# for name, param in model.named_parameters():
#     print(f"\nLayer: {name}")
#     print(f"Shape: {param.shape}")
#     print(f"Values:\n{param.data}")
# exit()

optimizer = create_optim(model, cfg_main.get_config()) 

trainer = Train_torch(cfg_main.get_config(),
                    model, 
                    optimizer, 
                    data_generator, 
                    loss_calculator,
                    test_data_generator,
                    calculate_l2_error,
                    vizualize)

trainer.train()
# trainer.printEval()