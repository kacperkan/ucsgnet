#!/usr/bin/env bash

dvc run -f dvc/cad_experiment/evaluate_model.dvc --no-exec  \
    -d ucsgnet/ucsgnet \
    -d ucsgnet/dataset.py \
    -d ucsgnet/utils.py \
    -d ucsgnet/common.py \
    -d data/cad \
    -d models/cad_main \
    -o paper-stuff/metrics/cad_experiment \
    python -m ucsgnet.ucsgnet.cad.eval_cad \
        --data_path data/cad/cad.h5 \
        --weights_path "models/cad_main/initial/ckpts/model.ckpt" \
        --out_dir paper-stuff/metrics/cad_experiment
