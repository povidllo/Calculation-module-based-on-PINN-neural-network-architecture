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
from Modules.allen_cahn.data_generator import data_generator
from Modules.allen_cahn.loss_calc import loss_calculator

import Modules.cfg_pinn_init as cfg_pinn_init
import Modules.cfg_opt_Adam_torch as cfg_opt_Adam_torch
import Modules.cfg_train_torch as cfg_train_torch

torch.manual_seed(44)
np.random.seed(44)
torch.cuda.manual_seed(44)

model = pinn(cfg_pinn_init.get_config())
# Вывод весов при инициализации
# for name, param in model.named_parameters():
#     print(f"\nLayer: {name}")
#     print(f"Shape: {param.shape}")
#     print(f"Values:\n{param.data}")

optimizer = create_optim(model, cfg_opt_Adam_torch.get_config()) 

trainer = Train_torch(cfg_train_torch.get_config(),
                      model, 
                      optimizer, 
                      data_generator, 
                      loss_calculator)
trainer.train()