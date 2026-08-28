model_name=meta-llama/Llama-3.2-3B-Instruct
ds_name=multiformat_knowedit_ZsRE

OMP_NUM_THREADS=1 python mf-memit/evaluate.py --alg_name baseline --model $model_name --ds_name $ds_name --use_cache --edit_formats ""