#!/usr/bin/env bash
dvc run --no-exec  \
    -d paper-stuff/metrics/cad_experiment \
    -d paper-stuff/2d-shapes-visualization/cad \
    -d paper-stuff/example_csg_on_binary_sets \
    -d paper-stuff/prediction-layer-example \
    -d paper-stuff/metrics/3d_experiment \
    -d data/3d_renders \
    -d ucsgnet/mesh_utils.c \
    echo "Congratulations, all experiments finished!"