import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()

    cfg.num_dots = [201, 513]

    return cfg
