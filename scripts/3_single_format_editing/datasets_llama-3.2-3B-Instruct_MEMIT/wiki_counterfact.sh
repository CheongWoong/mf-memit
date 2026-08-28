model_name=meta-llama/Llama-3.2-3B-Instruct
ds_name=multiformat_knowedit_wiki_counterfact

# Edit and evaluate using only the 'completion' format
OMP_NUM_THREADS=1 python -m mf-memit.evaluate \
    --alg_name MEMIT \
    --model $model_name \
    --ds_name $ds_name \
    --use_cache \
    --edit_formats completion