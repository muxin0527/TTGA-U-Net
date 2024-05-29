#!/bin/sh

DATASET_PATH=../DATASET

export PYTHONPATH=.././
export RESULTS_FOLDER=."$DATASET_PATH"/unetr_pp_trained_models
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task001_ACDC
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python3 /home/data/Program/unetr_plus_plus/unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_acdc 1 0
