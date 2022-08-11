#!/bin/bash

python3 -m autoopt.runner config/standard/schedulers_sgd_adagrad.yaml
python3 -m autoopt.runner config/standard/schedulers_aug_sgd_adagrad.yaml
