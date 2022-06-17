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
        elif isinstance(config[key], list):
            res.append(",".join([str(x) for x in config[key]]))
        else:
            res.append(config[key])

    return res


def get_path(config: Dict) -> str:
    data = [
        config['model'].get('print_name', config['model']['name']),
        config['dataset']['name'],
        'noaugment' if config['dataset'].get('noaugment', False) else 'augment',
        *stringify_config(config['optimizer']),
    ]
    if 'append_to_name' in config:
        data.append(config['append_to_name'])
    data.append(config['repeat'])
    return os.path.join(config['results_path'], '_'.join(map(str, data)))
