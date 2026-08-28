# Repository Guidelines

## Project Structure & Module Organization
Core editing code lives in `memit/`, `rome/`, `baselines/`, `dsets/`, and `util/`. Entry-point scripts for experiments are in `mf-memit/`, including `evaluate.py` and `causal_trace.py`. Reproducible shell workflows are grouped under `scripts/` by study stage, such as `scripts/1_causal_trace/` and `scripts/4_multi_format_editing/`. Datasets and caches live in `data/`, hyperparameters in `hparams/`, notebooks in `notebooks/`, and generated outputs in `results/` and `logs/`.

## Build, Test, and Development Commands
Set up the recommended environment with `bash scripts/installation/setup_conda.sh`; this creates the `mf-memit` Conda env from `scripts/installation/mf-memit.yml`. Optional data preparation scripts are:

- `bash scripts/installation/download_data.sh`
- `bash scripts/installation/generate_causal_tracing_data.sh`
- `bash scripts/installation/generate_multiformat_counterfact.sh`

Run causal tracing with `python mf-memit/causal_trace.py --model_name <hf-or-local-model> --fact_file data/causal_tracing_completion.json`. Collect layer statistics with `python -m rome.layer_stats --model_name <model> --layers 3,4,5`. Run evaluation with `python mf-memit/evaluate.py --alg_name MEMIT --model <model> --ds_name multiformat_counterfact_1000 --use_cache --edit_formats completion`.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, `snake_case` for functions and variables, and `UPPER_CASE` for module-level constants like `ALG_DICT`. Follow the existing style: small helper functions, direct `argparse` entry points, and minimal abstraction in experiment scripts. There is no repo-wide formatter config checked in, so keep changes consistent with nearby files and avoid unrelated refactors.

## Testing Guidelines
There is no dedicated `tests/` directory or enforced coverage target in the current tree. Validate changes by running the smallest relevant script or module for the area you touched, then record output paths under `results/` or `logs/`. For data-flow changes, prefer a narrow smoke test such as a single dataset or a limited layer range before launching full experiments.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase subjects such as `update fig`, `move`, and `add time tracing result`. Keep commit titles brief and imperative, but make them more specific when possible, for example `add llama eval output path`. Pull requests should state the experiment or code path changed, list affected scripts or hyperparameter files, mention required data/model dependencies, and include representative result files or plots when behavior changes.
