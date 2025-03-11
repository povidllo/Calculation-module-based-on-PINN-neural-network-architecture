import ml_collections
import torch

def get_config():
    cfg = ml_collections.ConfigDict()

    # initial parameters
    cfg.epochs = 10
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg.num_dots = [100, 256]
    
    #путь файла с правильными данными относительно директории проекта
    cfg.path_true_data = "/data/AC.mat"
    cfg.save_weights_path = "/ac_1d.pth"
    return cfg
