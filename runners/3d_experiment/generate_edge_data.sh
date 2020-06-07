#!/bin/bash

dvc run -f dvc/3d_experiment/generate_edge_data.dvc --no-exec  \
    -d ucsgnet/utils.py \
    -d ucsgnet/common.py \
    -d ucsgnet/generators/generate_edge_data_from_point.py \
    -d data/hdf5 \
    -d data/3d_reconstructions \
    -d data/pointcloud_surface \
    -o data/3d_edge_data \
    python -m ucsgnet.generators.generate_edge_data_from_point \
        --valid_shapes_file data/hdf5/all_vox256_img_test.txt \
        --ground_truth_folder data/pointcloud_surface \
        --reconstruction_folder data/3d_reconstructions \
        --out_dir data/3d_edge_data
