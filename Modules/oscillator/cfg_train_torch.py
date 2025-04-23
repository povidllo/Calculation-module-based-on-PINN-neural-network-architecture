import ml_collections
import torch

def get_config():
    cfg = ml_collections.ConfigDict()

    # initial parameters
    cfg.epochs = 10000
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg.num_dots = [400, 50]
    
    #путь файла с правильными данными относительно директории проекта
    cfg.path_true_data = "/data/OSC.npy"
    cfg.save_weights_path = "/osc_1d.pth"
    return cfg
