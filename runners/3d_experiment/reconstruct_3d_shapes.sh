#!/usr/bin/env bash

dvc run -f dvc/3d_experiment/reconstruct_3d_shapes.dvc --no-exec  \
    -d ucsgnet/ucsgnet \
    -d ucsgnet/dataset.py \
    -d ucsgnet/utils.py \
    -d data/hdf5 \
    -d ucsgnet/mesh_utils.c \
    -d models/3d_64 \
    -o data/3d_reconstructions \
    python -m ucsgnet.ucsgnet.reconstruct_3d_shapes \
        --valid all_vox256_img_test.hdf5 \
        --valid_shape_names all_vox256_img_test.txt \
        --processed data/hdf5 \
        --size 64 \
        --weights_path models/3d_64/initial/ckpts/model.ckpt \
        --sphere_complexity 3
