#!/bin/bash

python3 -m autoopt.runner config/experts/non_linearized_resnet34.yaml
python3 -m autoopt.runner config/experts/linearized_resnet34_3.yaml
python3 -m autoopt.runner config/experts/non_linearized_resnet34_2.yaml
