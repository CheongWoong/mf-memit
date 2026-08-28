#!/bin/bash

OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats completion TF MC triplet ODQA \
    --memit_merge