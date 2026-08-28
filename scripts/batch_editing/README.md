# MEMIT Batch Editing

`mf-memit/evaluate.py` supports joint MEMIT updates through `--num_edits`.
For example, the following command edits CounterFact-1K in batches of 100
facts using completion-only MEMIT:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mf-memit.evaluate \
  --alg_name MEMIT \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --ds_name multiformat_counterfact_1000 \
  --edit_formats completion \
  --num_edits 100 \
  --mom2_update_weight_override 1
```

For four-format MEMIT-XF, use the shared-value update with the same batch
size:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mf-memit.evaluate \
  --alg_name MEMIT-XF \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --ds_name multiformat_counterfact_1000 \
  --memit_merge \
  --edit_formats completion TF MC triplet \
  --num_edits 100 \
  --mom2_update_weight_override 1
```

The launcher reproduces the complete batch-size and covariance-weight sweep
used in the extended experiments. It schedules one run per available GPU:

```bash
GPU_IDS_STRING="0 1" \
  bash scripts/batch_editing/run_memit_batch_edit.sh
```

Set `MF_MEMIT_PYTHON` to select a Python executable. Logs are written to
`logs/memit_batch_edit/`, and per-case results are written to the ignored
`results/evaluation/` directory.
