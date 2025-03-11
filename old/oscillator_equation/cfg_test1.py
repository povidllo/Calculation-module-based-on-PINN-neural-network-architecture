import ml_collections



def get_config():
    cfg_name = "cfg_test1"
    cfg = ml_collections.ConfigDict()

    # wandb configuration
    cfg.project = "Oscillator"
    cfg.name = "test1"

    # initial parameters
    cfg.left_t = 0
    cfg.right_t = 1
    cfg.num_t = 500
    cfg.num_ph = 30
    cfg.epochs = 500
    cfg.hidden_sizes = [64, 64, 64]
    cfg.lr = 1e-4

    cfg.save_path = "./weights/"+cfg_name+".pth"
    #data
    cfg.save_res_img = "./imgs/"+cfg_name+"_res"
    cfg.save_loss_img = "./imgs/"+cfg_name+"_loss"
    cfg.save_l2_img = "./imgs/"+cfg_name+"_l2"

    #Fourier
    cfg.Fourier = False

    return cfg  
