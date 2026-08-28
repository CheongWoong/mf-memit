# MF-MEMIT: Multi-Format MEMIT

This repository evaluates whether parametric knowledge edits generalize across
heterogeneous task formats and implements multi-format editing with a shared
value objective. The core release reproduces cross-format evaluation and
MEMIT-XF on CounterFact-1K; paper-specific plotting and rebuttal artifacts are
not required.

## Installation

Create the Python 3.10 Conda environment:

```bash
bash scripts/installation/setup_conda.sh
conda activate mf-memit
```

The main experiment uses `meta-llama/Llama-3.2-3B-Instruct`, so Hugging Face
access to the gated model and a CUDA GPU are required. The released
`data/multiformat_counterfact_1000.json` already contains the six evaluation
formats used below.

## Core Reproduction

First collect the second-moment statistics for the six edited MLP layers. The
statistics are cached under `data/stats/` and reused by later runs.

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/2_memit_collect_layer_stats/llama-3.2-3B-Instruct/collect_layer_stats.sh
```

Run the unedited baseline, completion-only MEMIT, and six-format MEMIT-XF:

```bash
# Pre-edit cross-format evaluation
python -m mf-memit.evaluate \
  --alg_name baseline \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --ds_name multiformat_counterfact_1000 \
  --edit_formats ""

# Single-format editing
python -m mf-memit.evaluate \
  --alg_name MEMIT \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --ds_name multiformat_counterfact_1000 \
  --use_cache \
  --edit_formats completion

# Multi-format editing with one shared value per fact
python -m mf-memit.evaluate \
  --alg_name MEMIT \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --ds_name multiformat_counterfact_1000 \
  --use_cache \
  --edit_formats completion triplet ODQA MC TF YN \
  --memit_merge
```

Per-case predictions and metrics are written as JSONL files under
`results/evaluation/meta-llama_Llama-3.2-3B-Instruct/`. These outputs contain
the values needed to aggregate efficacy, cross-format generalization,
consistency, and specificity.

Equivalent launchers are available in `scripts/3_single_format_editing/` and
`scripts/4_multi_format_editing/`. Dataset regeneration is optional:

```bash
bash scripts/installation/download_data.sh
bash scripts/installation/generate_multiformat_counterfact.sh
```

MEMIT batch editing is enabled with `--num_edits`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mf-memit.evaluate \
  --alg_name MEMIT \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --ds_name multiformat_counterfact_1000 \
  --edit_formats completion \
  --num_edits 100 \
  --mom2_update_weight_override 1
```

The complete batch and sequential workflows are documented separately:

- [MEMIT batch editing](scripts/batch_editing/README.md)
- [AlphaEdit sequential editing](scripts/sequential_editing/README.md)

## Optional Analyses

Causal-tracing inputs and scripts are included under `data/`,
`scripts/1_causal_trace/`, and `mf-memit/causal_trace.py`. AlphaEdit-based
baselines and the sequential-editing extension additionally require an
EasyEdit checkout at `../EasyEdit`; they are not dependencies of the core
MEMIT-XF reproduction.

## References

- [Original MEMIT implementation](https://github.com/kmeng01/memit)
- [Original MEMIT-Merge implementation](https://github.com/NUSTM/MEMIT-Merge)
- [EasyEdit](https://github.com/zjunlp/EasyEdit)
