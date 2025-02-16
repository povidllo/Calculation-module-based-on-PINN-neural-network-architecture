from typing import Optional, List
from beanie import Document

class HyperParam(Document):
  hidden_count: int
  input_dim: int
  output_dim: int
  hidden_sizes: List[int]
  Fourier: bool
  FinputDim: Optional[int]
  FourierScale: Optional[int]
  
  lr: float
  betas: tuple[float, float]
  
  epochs: int
  device: str
  num_dots: List[int]
  path_true_data: str
  save_weights_path: str


class Weights(Document):
  save_weights_path: str
  #можно добавить loss


class NeuralNetwork(Document):
  name: str
  equation: str
  weights: Weights
  hyper_params: HyperParam
