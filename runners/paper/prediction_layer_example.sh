#!/usr/bin/env bash

dvc run -f dvc/paper/prediction_layer_example.dvc --no-exec \
    -d ucsgnet/visualization \
    -d ucsgnet/ucsgnet/shape_evaluators.py \
    -d ucsgnet/visualization/generate_example_csg_layer.py \
    -o paper-stuff/prediction-layer-example \
    python -m ucsgnet.visualization.generate_example_csg_layer \
        --out_path paper-stuff/prediction-layer-example
