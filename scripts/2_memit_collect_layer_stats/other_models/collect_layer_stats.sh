model_name=$1
layers=$2

OMP_NUM_THREADS=1 python -m rome.layer_stats --model_name $model_name --layers $layers --sample_size 100000