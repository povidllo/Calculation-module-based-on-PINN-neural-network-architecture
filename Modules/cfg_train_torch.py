import ml_collections
import torch

def get_config():
    cfg = ml_collections.ConfigDict()

    # initial parameters
    cfg.epochs = 11
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg.left_t = 0
    cfg.right_t = 1
    cfg.num_t = 100
    cfg.left_x = -1
    cfg.right_x = 1
    cfg.num_x = 256

    return cfg  # Убедитесь, что возвращаете cfg
