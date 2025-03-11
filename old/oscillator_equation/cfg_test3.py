import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg_name = "cfg_test3"

    # wandb configuration
    cfg.project = "Oscillator"
    cfg.name = "test3"

    # initial parameters
    cfg.left_t = 0
    cfg.right_t = 1
    cfg.num_t = 400
    cfg.num_ph = 50
    cfg.epochs = 10000
    cfg.hidden_sizes = [32, 32, 32]
    cfg.lr = 1e-4

    cfg.save_path = "./weights/test3.pth"
    #data   
    cfg.save_res_img = "./imgs/"+cfg_name+"_res"
    cfg.save_loss_img = "./imgs/"+cfg_name+"_loss"
    cfg.save_l2_img = "./imgs/"+cfg_name+"_l2"

    #Fourier
    cfg.Fourier = True
    cfg.FourierType = "basic"
    cfg.FinputDim = 256
    cfg.FourierScale = 1


    return cfg  
