#!/usr/bin/env bash
dvc run --no-exec -f dvc/initialize_env.dvc \
    -d ucsgnet/mesh_utils.pyx \
    -o ucsgnet/mesh_utils.c \
    python setup.py build_ext --inplace
