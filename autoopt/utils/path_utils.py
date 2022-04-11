import os
from typing import Dict


def stringify_config(config: Dict):
    keys = sorted(list(config.keys()))
    if "name" in keys:
        keys.remove("name")
        keys = ["name"] + keys
    res = []
    for key in keys:
        if isinstance(config[key], Dict):
            res.extend(stringify_config(config[key]))
        else:
            res.append(config[key])

    return res


def get_path(config: Dict) -> str:
    data = [
        config['model'].get('print_name', config['model']['name']),
        config['dataset']['name'],
        *stringify_config(config['optimizer']),
        config['repeat']
    ]
    return os.path.join(config['results_path'], '_'.join(map(str, data)))
