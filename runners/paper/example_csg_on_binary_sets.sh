#!/usr/bin/env bash

dvc run -f dvc/paper/example_csg_on_binary_sets.dvc --no-exec \
    -d ucsgnet/visualization \
    -d ucsgnet/ucsgnet/shape_evaluators.py \
    -o paper-stuff/example_csg_on_binary_sets \
    python -m ucsgnet.visualization.visualize_csg_on_binary_sets \
        --out_dir paper-stuff/example_csg_on_binary_sets
