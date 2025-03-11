import ml_collections
import torch

def get_config():
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
    cfg.epochs = 450
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.num_dots = [400, 50]
    
    #путь файла с данными относительно директории проекта
    cfg.path_true_data = "/data/OSC.npy"
    cfg.save_weights_path = "/osc_1d.pth"

    return cfg