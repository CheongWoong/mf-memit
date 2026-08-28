model_name=$1

OMP_NUM_THREADS=1 python mf-memit/causal_trace.py --model_name $model_name --fact_file data/causal_tracing_completion.json