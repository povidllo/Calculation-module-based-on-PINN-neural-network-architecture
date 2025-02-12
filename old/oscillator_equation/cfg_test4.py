import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg_name = "cfg_test4"

    # wandb configuration
    cfg.project = "Oscillator"
    cfg.name = "test4"

    # initial parameters
    cfg.left_t = 0
    cfg.right_t = 1
    cfg.num_t = 500
    cfg.num_ph = 20
    cfg.epochs = 20000
    cfg.hidden_sizes = [128, 128, 128]
    cfg.lr = 1e-3

    cfg.save_path = "./weights/cfg_test4.pth"
    #data
    cfg.save_res_img = "./imgs/"+cfg_name+"_res"
    cfg.save_loss_img = "./imgs/"+cfg_name+"_loss"
    cfg.save_l2_img = "./imgs/"+cfg_name+"_l2"


    #Fourier
    cfg.Fourier = False

    return cfg  
