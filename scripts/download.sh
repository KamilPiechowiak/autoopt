#!/bin/bash

gsutil -m rsync -x ".*pt" -r gs://kamil-piechowiak-weights-transfer/autoopt/ results