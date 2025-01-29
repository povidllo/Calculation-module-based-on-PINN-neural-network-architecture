import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()

    # initial parameters
    cfg.hidden_count = 128
    
    cfg.input_dim = 2
    cfg.output_dim = 1
    cfg.hidden_sizes = [128, 128, 128, 128]

    #Fourier
    cfg.Fourier = False

    return cfg  # Убедитесь, что возвращаете cfg
