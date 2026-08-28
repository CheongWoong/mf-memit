#!/bin/bash

# Edit and evaluate using only the 'MC' (Multiple Choice) format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats MC

# Edit and evaluate using only the 'TF' (True/False) format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats TF

# Edit and evaluate using only the 'triplet' format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats triplet

# Edit and evaluate using only the 'ODQA' format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats ODQA

# Edit and evaluate using only the 'YN' format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats YN