#!/usr/bin/env bash

dvc run -f dvc/3d_experiment/run_experiment.dvc --no-exec  \
    -d ucsgnet/ucsgnet \
    -d ucsgnet/dataset.py \
    -d ucsgnet/utils.py \
    -d ucsgnet/loggers.py \
    -d ucsgnet/common.py \
    -d data/hdf5 \
    -o models/3d_64 \
    python -m ucsgnet.ucsgnet.train_3d \
        --train all_vox256_img_train.hdf5 \
        --valid all_vox256_img_test.hdf5 \
        --processed data/hdf5 \
        --experiment_name 3d \
        --lr 0.0001 \
        --batch_size 16 \
        --num_dimensions 3 \
        --shapes_per_type 32 \
        --out_shapes_per_layer 12 \
        --beta1 0.5 \
        --beta2 0.99 \
        --seed 0 \
        --num_csg_layers 5 \
        --full_resolution_only
