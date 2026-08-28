# Edit and evaluate using only the 'completion' format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate_baselines_with_easyedit \
    --alg_name PMET \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ds_name multiformat_counterfact_1000 \
    --use_cache \
    --edit_formats completion