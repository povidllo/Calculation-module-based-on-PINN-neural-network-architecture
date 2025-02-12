from typing import Optional, List
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


class Weight(Document):
  file_path: str