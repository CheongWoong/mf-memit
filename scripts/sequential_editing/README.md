# AlphaEdit Sequential Editing

This experiment follows the AlphaEdit sequential protocol: jointly edit 100
facts per round, retain the model update and edited-key covariance across 10
rounds, and evaluate all 1,000 facts after the final round.

Each fact can use one or more source formats. Multi-format runs optimize one
shared value per fact and solve all 100 factual associations in a joint update.
`--fact_chunk_size` only bounds activation memory; it does not change the batch
objective.

The experiment requires an EasyEdit checkout at `../EasyEdit`. By default, the
launcher maps K=1, K=4, and K=6 to GPUs 1, 2, and 3. Override the mapping with
`K1_GPU`, `K4_GPU`, and `K6_GPU`, and set `EASYEDIT_PYTHON` when needed:

```bash
K1_GPU=0 K4_GPU=1 K6_GPU=2 EASYEDIT_PYTHON=/path/to/python \
  bash scripts/sequential_editing/run_alphaedit_batch_sequential.sh
```

For one custom run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mf-memit.evaluate_alphaedit_batch_sequential \
  --case_limit 1000 \
  --batch_size 100 \
  --fact_chunk_size 2 \
  --edit_formats completion TF MC triplet \
  --run_name alphaedit_k4_batch100_seq1000 \
  --resume
```

Detailed rows, checkpoints, and logs are written under the ignored `results/`
and `logs/` directories.
