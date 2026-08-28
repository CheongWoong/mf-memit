#!/bin/bash

model_name=$1

# Edit and evaluate using only the 'completion' format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model $model_name \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats completion triplet ODQA MC TF YN \
    --memit_merge
