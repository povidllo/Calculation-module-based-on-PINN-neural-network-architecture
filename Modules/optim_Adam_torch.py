# class Opt_Adam_torch:
#     def __init__(self,
#                  model,
#                  cfg):
import torch
def create_optim(model, cfg):
    return torch.optim.Adam(model.parameters(), betas=cfg.betas, lr=cfg.lr)