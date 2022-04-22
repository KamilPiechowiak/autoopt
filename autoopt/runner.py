import argparse
from typing import Dict
import logging
import copy

from autoopt.distributed.base_connector import BaseConnector
from autoopt.utils import read_yaml
from autoopt.training import train


def get_device_connector(config: Dict) -> BaseConnector:
    if config['tpu']:
        from autoopt.distributed.xla_connector import XlaConnector
        return XlaConnector()
    else:
        from autoopt.distributed.simple_connector import SimpleConnector
        return SimpleConnector()


def run_experiments(rank: int, config: Dict, connector: BaseConnector, log: str):
    logging.basicConfig(format=f"{rank}:%(levelname)s:%(message)s", level={
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "error": logging.ERROR,
    }[log.lower()])
    for experiment in config["experiments"]:
        experiment_config = {
            **copy.deepcopy(config["general"]),
            **copy.deepcopy(experiment.copy())
        }
        repeats_start = experiment_config.get("repeats_start", 0)
        for repeat in range(repeats_start, repeats_start + experiment_config["repeats"]):
            train({**copy.deepcopy(experiment_config), "repeat": repeat}, connector)


def run(args: Dict):
    config = read_yaml(args.config_path)
    connector = get_device_connector(config['general'])

    connector.run(run_experiments, args=(config, connector, args.log),
                  nprocs=config["general"].get('num_cores'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("--log", type=str, help="Set logging level", default="info")
    args = parser.parse_args()
    run(args)
