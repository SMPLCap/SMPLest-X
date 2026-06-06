#!/usr/bin/env bash

JOB_NAME=$1
GPUS=$2
CONFIG=$3

PYTHONPATH=../:$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --nproc_per_node $GPUS \
    --master_port 29500 main/train.py \
        --num_gpus $GPUS \
        --exp_name train_$JOB_NAME \
        --master_port 29500 \
        --config $CONFIG
