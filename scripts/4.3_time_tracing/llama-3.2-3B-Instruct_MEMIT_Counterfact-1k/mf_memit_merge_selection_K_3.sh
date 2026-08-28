#!/bin/bash

OMP_NUM_THREADS=1 python -m mf-memit.evaluate_time_tracing \
    --alg_name MEMIT \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --edit_formats completion TF MC \
    --memit_merge