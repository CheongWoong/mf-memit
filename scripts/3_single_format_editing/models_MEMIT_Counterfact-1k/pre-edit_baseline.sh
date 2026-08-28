model_name=$1

OMP_NUM_THREADS=1 python mf-memit/evaluate.py --alg_name baseline --model $model_name --ds_name multiformat_counterfact_1000 --use_cache --edit_formats ""