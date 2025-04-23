import ml_collections

def get_config():
    cfg_name = "cfg_test2"
    cfg = ml_collections.ConfigDict()

    # wandb configuration
    cfg.project = "Oscillator"
    cfg.name = "test2"

    # initial parameters
    cfg.left_t = 0
    cfg.right_t = 1
    cfg.num_t = 500
    cfg.num_ph = 30
    cfg.epochs = 20000
    cfg.hidden_sizes = [32, 32, 32, 32]
    cfg.lr = 1e-4

    cfg.save_path = "./weights/cfg_test2.pth"
    #data
    cfg.save_res_img = "./imgs/"+cfg_name+"_res"
    cfg.save_loss_img = "./imgs/"+cfg_name+"_loss"
    cfg.save_l2_img = "./imgs/"+cfg_name+"_l2"


    #Fourier
    cfg.Fourier = False

    return cfg  
