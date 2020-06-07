#!/bin/bash

dvc run -f dvc/3d_experiment/evaluate_model.dvc --no-exec  \
    -d ucsgnet/ucsgnet \
    -d ucsgnet/utils.py \
    -d ucsgnet/common.py \
    -d data/hdf5 \
    -d data/3d_reconstructions \
    -d data/shapenet \
    -d data/pointcloud_surface \
    -d data/3d_edge_data \
    -o paper-stuff/metrics/3d_experiment \
    python -m ucsgnet.ucsgnet.evaluate_on_3d_data \
        --valid_shape_names_file data/hdf5/all_vox256_img_test.txt \
        --reconstructed_shapes_folder data/3d_reconstructions \
        --raw_shapenet_data_folder data/shapenet \
        --ground_truth_point_surface data/pointcloud_surface \
        --edges_data_folder data/3d_edge_data \
        --out_folder paper-stuff/metrics/3d_experiment
