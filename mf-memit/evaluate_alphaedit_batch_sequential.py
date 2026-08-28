import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EASYEDIT_ROOT = ROOT.parent / "EasyEdit"
sys.path.insert(0, str(EASYEDIT_ROOT))

from easyeditor import AlphaEditHyperParams, BaseEditor  # noqa: E402

from baselines.alphaedit_sequential import SequentialAlphaEdit  # noqa: E402
from .alphaedit_sequential_eval_utils import (  # noqa: E402
    EDIT_FIELDS,
    build_fact_requests,
    evaluate_cases,
    summarize,
)


def metadata(args):
    return {
        "model": args.model,
        "dataset": args.ds_name,
        "case_limit": args.case_limit,
        "batch_size": args.batch_size,
        "edit_formats": args.edit_formats,
        "run_name": args.run_name,
    }


def atomic_json_dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def save_checkpoint(path, args, completed_cases, edit_seconds, editor, model):
    payload = {
        "metadata": metadata(args),
        "completed_cases": completed_cases,
        "edit_seconds": edit_seconds,
        "alphaedit_state": editor.state_dict(model),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def write_rows(path, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--ds_name", default="multiformat_counterfact_1000")
    parser.add_argument("--case_limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--fact_chunk_size", type=int, default=10)
    parser.add_argument(
        "--edit_formats",
        nargs="+",
        choices=list(EDIT_FIELDS),
        default=["completion"],
    )
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", default="results/alphaedit_batch_sequential")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.case_limit < 1 or args.batch_size < 1 or args.fact_chunk_size < 1:
        raise ValueError("case_limit, batch_size, and fact_chunk_size must be positive.")
    if args.case_limit % args.batch_size != 0:
        raise ValueError("case_limit must be divisible by batch_size.")

    output_dir = ROOT / args.output_dir
    rows_path = output_dir / f"{args.run_name}.jsonl"
    summary_path = output_dir / f"{args.run_name}.summary.json"
    progress_path = output_dir / "progress" / f"{args.run_name}.json"
    checkpoint_path = output_dir / "checkpoints" / f"{args.run_name}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and summary_path.exists():
        print(f"Run already complete: {summary_path}", flush=True)
        return
    if not args.resume and (rows_path.exists() or checkpoint_path.exists()):
        raise FileExistsError(f"Output exists for {args.run_name}; use --resume.")

    with open(ROOT / "data" / f"{args.ds_name}.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)[: args.case_limit]
    if len(data) != args.case_limit:
        raise ValueError(f"Requested {args.case_limit} cases, found {len(data)}.")

    num_rounds = args.case_limit // args.batch_size
    if args.dry_run:
        print(
            json.dumps(
                {
                    "total_cases": len(data),
                    "batch_size": args.batch_size,
                    "num_rounds": num_rounds,
                    "formats_per_fact": len(args.edit_formats),
                    "evaluate_after_all_rounds": True,
                },
                sort_keys=True,
            )
        )
        return

    hparams_path = ROOT / "hparams/EasyEdit/AlphaEdit" / f"{args.model.split('/')[-1]}.yaml"
    hparams = AlphaEditHyperParams.from_hparams(str(hparams_path))
    base_editor = BaseEditor.from_hparams(hparams)
    model, tok = base_editor.model, base_editor.tok
    model.eval()
    tok.pad_token = tok.eos_token
    device = torch.device(f"cuda:{hparams.device}")

    editor = SequentialAlphaEdit(hparams, fact_chunk_size=args.fact_chunk_size)
    editor.reset_chain(model)
    pristine_weights = editor.snapshot_weights(model)
    completed_cases = 0
    edit_seconds = 0.0

    if args.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        if payload.get("metadata") != metadata(args):
            raise ValueError(f"Checkpoint metadata mismatch for {checkpoint_path}.")
        completed_cases = int(payload["completed_cases"])
        edit_seconds = float(payload.get("edit_seconds", 0.0))
        if completed_cases % args.batch_size != 0:
            raise ValueError("Checkpoint does not end at a batch boundary.")
        editor.load_state_dict(model, payload["alphaedit_state"])
        print(
            f"BATCH_SEQUENTIAL_RESUME run={args.run_name} "
            f"completed={completed_cases}/{args.case_limit}",
            flush=True,
        )

    try:
        for batch_start in range(completed_cases, args.case_limit, args.batch_size):
            batch = data[batch_start : batch_start + args.batch_size]
            fact_requests = [
                build_fact_requests(item, item["case_id"], args.edit_formats)
                for item in batch
            ]
            started_at = time.time()
            editor.apply_batch(model, tok, fact_requests)
            edit_seconds += time.time() - started_at
            completed_cases = batch_start + len(batch)
            save_checkpoint(
                checkpoint_path,
                args,
                completed_cases,
                edit_seconds,
                editor,
                model,
            )
            atomic_json_dump(
                {
                    "metadata": metadata(args),
                    "completed_cases": completed_cases,
                    "completed_rounds": completed_cases // args.batch_size,
                    "edit_seconds": edit_seconds,
                },
                progress_path,
            )
            print(
                f"BATCH_SEQUENTIAL_ROUND_DONE run={args.run_name} "
                f"round={completed_cases // args.batch_size}/{num_rounds} "
                f"completed={completed_cases}/{args.case_limit}",
                flush=True,
            )

        eval_started_at = time.time()
        rows = evaluate_cases(data, model, tok, device)
        eval_seconds = time.time() - eval_started_at
        summary = summarize(rows)
        summary.update(
            {
                "run_name": args.run_name,
                "model": args.model,
                "dataset": args.ds_name,
                "num_evaluated_cases": len(rows),
                "batch_size": args.batch_size,
                "num_rounds": num_rounds,
                "edit_formats": args.edit_formats,
                "edit_seconds": edit_seconds,
                "edit_seconds_per_fact": edit_seconds / len(data),
                "eval_seconds": eval_seconds,
            }
        )
        write_rows(rows_path, rows)
        atomic_json_dump(summary, summary_path)
        checkpoint_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        print("BATCH_SEQUENTIAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    finally:
        editor.restore_weights(model, pristine_weights)
        editor.reset_chain(model)


if __name__ == "__main__":
    main()
