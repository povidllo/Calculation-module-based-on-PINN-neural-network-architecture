from typing import Optional, List, Callable
from beanie import Document

class HyperParam(Document):
  id: int
  power_time_vector: int
  power_dom_vector: int
  layers_count: int
  hidden_sizes: List[int]
  out_power: int
  batch_size: int
  epochs: int
  fourier: int
  fourier_type: Optional[str]
  fourier_dim: Optional[int]
  fourier_scale: Optional[int]

class DataSet(Document):
    power_time_vector: int
    power_dom_vector: int
    param: List[int]
    data_generator: Callable
    loss_calc: Callable

class NeuralNetwork(Document):
  hyperparam_id: int
  weight_id: int

  init: Callable
  inference: Callable

class Weight(Document):
  weights: List[int]

class Optimizer(Document):
  data_set_id: int
  neural_network_id: int
  train: Callable
  finetune: Callable
  save_weights: Callable
  show_res: Callable
