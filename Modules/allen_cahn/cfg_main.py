import ml_collections
import torch

def get_config():
    cfg = ml_collections.ConfigDict()

    # initial parameters for optimizers
    optimizers = ["Adam", "NAdam", "LBFGS"]
    cfg.optimizer = optimizers[0]
    cfg.lr = 1e-3
    cfg.betas = (0.999, 0.999)

    # initial parameters for nn
    cfg.hidden_count = 128
    
    cfg.input_dim = 2
    cfg.output_dim = 1
    cfg.hidden_sizes = [128, 128, 128, 128]

    #Fourier
    cfg.Fourier = False
    cfg.FinputDim = None
    cfg.FourierScale = None

    #Test dataset
    cfg.num_dots = [201, 513]

    #Train
    cfg.epochs = 10
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.num_dots = [100, 256]
    
    #путь файла с правильными данными относительно директории проекта
    cfg.path_true_data = "/data/AC.mat"
    cfg.save_weights_path = "/ac_1d.pth"

    return cfg 
