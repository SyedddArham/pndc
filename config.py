import yaml
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    input_dir: str = "input"
    output_dir: str = "results"
    image_sizes: List[Tuple[int, int]] = None
    operators: List[str] = None
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.image_sizes is None:
            self.image_sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        if self.operators is None:
            self.operators = ['sobel', 'prewitt', 'laplacian']

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict) 