from typing import Dict
import yaml

def read_yaml(path: str) -> Dict:
    with open(path) as file:
        return yaml.full_load(file)
