#!/bin/bash

ZONE=us-central1-f
TPU_INSTANCE_NAME=tpu

gcloud alpha compute tpus tpu-vm create $TPU_INSTANCE_NAME \
--zone=$ZONE \
--accelerator-type=v2-8 \
--version=v2-alpha

zip -r autoopt.zip autoopt config scripts -x *__pycache__*
gcloud alpha compute tpus tpu-vm scp autoopt.zip $TPU_INSTANCE_NAME:. --zone=$ZONE

gcloud alpha compute tpus tpu-vm ssh $TPU_INSTANCE_NAME --zone=$ZONE

#vm begin
gcloud auth login
unzip -oq autoopt.zip
mkdir results
mkdir data

pip3 install timm tqdm matplotlib

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PT_XLA_DEBUG=1


python3 -m autoopt.runner config/lr_test.yaml
python3 -m autoopt.runner config/lr.yaml
nohup python3 -m autoopt.runner config/lr_resnet.yaml 2>&1 > log.log &

gcloud alpha compute tpus tpu-vm delete $TPU_INSTANCE_NAME --zone=$ZONE

gcloud alpha compute tpus tpu-vm scp --recurse --scp-flag="--exclude='*.pt'" $TPU_INSTANCE_NAME:results results --zone=$ZONE

gcloud alpha compute tpus tpu-vm ssh $TPU_INSTANCE_NAME --zone=$ZONE -- -L 6007:localhost:6006