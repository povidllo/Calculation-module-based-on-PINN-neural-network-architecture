import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()

    # initial parameters
    cfg.lr = 1e-4
    cfg.betas=(0.9, 0.999)

    return cfg  # Убедитесь, что возвращаете cfg
