#!/usr/bin/env bash

dvc run -f dvc/cad_experiment/run_experiment.dvc --no-exec  \
    -d ucsgnet/ucsgnet \
    -d ucsgnet/dataset.py \
    -d ucsgnet/utils.py \
    -d ucsgnet/loggers.py \
    -d ucsgnet/common.py \
    -d data/cad \
    -o models/cad_main \
    python -m ucsgnet.ucsgnet.cad.train_cad \
        --data_path data/cad/cad.h5 \
        --experiment_name cad \
        --lr 0.0001 \
        --batch_size 16 \
        --num_dimensions 2 \
        --shapes_per_type 16 \
        --out_shapes_per_layer 4 \
        --beta1 0.5 \
        --beta2 0.99 \
        --num_csg_layers 2
