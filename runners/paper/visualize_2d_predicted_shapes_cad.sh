#!/bin/bash

dvc run -f dvc/paper/visualize_2d_predicted_shapes_cad.dvc --no-exec \
    -d models/cad_main \
    -d ucsgnet/ucsgnet \
    -d ucsgnet/dataset.py \
    -d data/cad \
    -o paper-stuff/2d-shapes-visualization/cad \
    python -m ucsgnet.ucsgnet.visualize_2d_predicted_shapes \
        --ckpt models/cad_main/initial/ckpts/model.ckpt \
        --valid cad.h5 \
        --data data/cad \
        --data_type cad
