ds_name="multiformat_counterfact_1000"
for model in meta-llama/Llama-3.2-3B-Instruct; do
    for format in completion triplet ODQA MC TF YN; do
        OMP_NUM_THREADS=1 python mf-memit/save_v_star_trajectory.py --alg_name MEMIT --model $model --ds_name $ds_name --use_cache --edit_formats $format
    done
done