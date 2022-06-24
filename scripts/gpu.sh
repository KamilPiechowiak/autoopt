#!/bin/bash

NAME=vm
ZONE=us-central1-f
MACHINE_TYPE=n1-standard-2


gcloud compute instances create $NAME \
    --machine-type $MACHINE_TYPE \
    --zone $ZONE \
    --boot-disk-size=120GB \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy TERMINATE --restart-on-failure

zip -r autoopt.zip autoopt config scripts -x *__pycache__*
gcloud compute scp autoopt.zip $NAME:. --zone=$ZONE

gcloud compute ssh $NAME --zone=$ZONE

#vm begin
gcloud auth login
unzip -oq autoopt.zip
mkdir results data

pip3 install timm tqdm matplotlib tensorboard

python3 -m autoopt.runner config/lr_test.yaml
python3 -m autoopt.runner config/lr.yaml
nohup python3 -m autoopt.runner config/experts/linearized_adam.yaml 2>&1 > log.log &

lr_resnet_noaug_normalize
lr_resnet_augment_normalized
standard/schedulers
standard/schedulers_augment
experts/linearized_analytical
experts/linearized_normalized
experts/non_linearized_independent_pretrain
prof

experts/linearized_adam
prof_augment
experts/linearize_adam_full

gcloud compute scp --recurse --scp-flag="--exclude='*.pt'" $NAME:results results --zone=$ZONE

gcloud compute ssh $NAME --zone=$ZONE -- -L 6007:localhost:6006

gsutil cp -r gs://autoopt/SmallResnet20_CIFAR10_noaugment_Adam_* results