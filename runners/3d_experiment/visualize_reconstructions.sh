#!/bin/bash

dvc run -f dvc/3d_experiment/visualize_reconstructions.dvc --no-exec  \
    -d ucsgnet/visualization/visualize_shapes.py \
    -d data/3d_reconstructions \
    -d data/shapenet \
    -o data/3d_renders \
    python -m ucsgnet.visualization.visualize_shapes \
        --in_folder data/3d_reconstructions \
        --out_folder data/3d_renders \
        --raw_folder data/shapenet
