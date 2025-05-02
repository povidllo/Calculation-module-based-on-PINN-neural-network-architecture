import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()
    print(type(cfg))

    # wandb configuration
    cfg.project = "Allen-cahn"
    cfg.name = "name"

    # initial parameters
    cfg.left_t = 0
    cfg.right_t = 1
    cfg.num_t = 100
    cfg.left_x = -1
    cfg.right_x = 1
    cfg.num_x = 256
    cfg.epochs = 11
    cfg.hidden_count = 128
    cfg.lr = 1e-3

    cfg.input_dim = 2
    cfg.output_dim = 1
    cfg.hidden_sizes = [128, 128, 128, 128]

    #data
    cfg.data_path= "./data/AC.mat"    

    #Fourier
    cfg.Fourier = False

    return cfg 
