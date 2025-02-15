import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()

    # initial parameters
    cfg.hidden_count = 32
    
    cfg.input_dim = 1
    cfg.output_dim = 1
    cfg.hidden_sizes = [32, 32, 32]

    #Fourier
    cfg.Fourier = False
    cfg.FinputDim = None
    cfg.FourierScale = None

    return cfg
