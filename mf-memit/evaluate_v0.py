import time
import os
from copy import deepcopy
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}


def get_avg_logprob(prompt: str, target: str, tokenizer, model, device):
    """
    Computes the average log probability of target given prompt.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with tokenizer.as_target_tokenizer():
        target_enc = tokenizer(" " + target, return_tensors="pt", add_special_tokens=False).to(device)

    full_input_ids = torch.cat([enc.input_ids, target_enc.input_ids], dim=1)

    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = full_input_ids[:, 1:]

    log_probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Get top-1 predicted token for first target token
    input_len = enc.input_ids.shape[1]
    top_token_id = log_probs[0, input_len - 1].argmax().item()
    predicted_token = tokenizer.decode(top_token_id)

    avg_logprob = token_log_probs[0, input_len - 1:].mean().item()
    return avg_logprob, predicted_token

def compute_efficacy(entry, tokenizer, model, device):
    """
    Computes efficacy score and magnitude for one edit entry (new vs true target).
    """
    if "{}" in entry["prompt"]:
        prompt = entry["prompt"].format(entry["subject"])
    else:
        prompt = entry["prompt"]
    target_new = entry["target_new"]["str"]
    target_true = entry["target_true"]["str"]

    if target_true == "":
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(enc.input_ids)
            logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_token_id = probs.argmax(dim=-1).item()
        predicted_token = tokenizer.decode(top_token_id)
        prob = probs[0, top_token_id].item()
        score = 1 if target_new.lower().startswith(predicted_token.strip().lower()) else 0
        return score, prob, predicted_token

    lp_new, _ = get_avg_logprob(prompt, target_new, tokenizer, model, device)
    lp_true, predicted_token = get_avg_logprob(prompt, target_true, tokenizer, model, device)
    delta = lp_new - lp_true
    score = 1 if delta > 0 else 0
    return score, delta, predicted_token

def compute_specificity(entry, tokenizer, original_model, edited_model, device):
    """
    Computes specificity score and magnitude for one edit entry by comparing model outputs before and after editing.
    The new specificity metric measures whether the model's generation on a neighborhood prompt is less likely to be the original generation (before edit) and more likely to be the new target.
    """
    target_new = entry["target_new"]["str"]
    target_true = entry["target_true"]["str"]
    
    scores = []
    deltas = []

    if "neighborhood_prompts" not in entry or len(entry["neighborhood_prompts"]) < 1:
        return -1, -1

    if filter_correct := False:
        with tokenizer.as_target_tokenizer():
            target_true_enc = tokenizer(" " + target_true, return_tensors="pt", add_special_tokens=False).to(device)
        
        target_true_first_token_id = target_true_enc.input_ids[0, 0].item()

    for prompt in entry["neighborhood_prompts"]:
        if filter_correct := False:
            # Get top-1 token from original model
            original_enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                original_outputs = original_model(original_enc.input_ids)
                original_logits = original_outputs.logits
            
            original_probs = torch.nn.functional.softmax(original_logits[:, -1, :], dim=-1)
            original_top_token_id = original_probs.argmax().item()

            # Filter for prompts that generate target_true
            if original_top_token_id != target_true_first_token_id:
                continue

        # Get average log probabilities from the edited model
        lp_true, _ = get_avg_logprob(prompt, target_true, tokenizer, edited_model, device)
        lp_new, _ = get_avg_logprob(prompt, target_new, tokenizer, edited_model, device)

        delta = lp_true - lp_new
        score = 1 if delta > 0 else 0
        scores.append(score)
        deltas.append(delta)

    avg_score = sum(scores) / len(scores)
    avg_delta = sum(deltas) / len(deltas) if deltas else 0
    return avg_score, avg_delta


def evaluate_all_per_case(ds_name, model_name, alg_name, use_cache, edit_formats=None, memit_merge=False, single_key=False, single_value=False):
    """
    Evaluates each case (question) across all 3 formats, and writes jsonl output.
    """
    if edit_formats is None:
        edit_formats = []
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT.get(alg_name, (None, lambda *x, **kwargs: (x[0], None)))

    # Get run hyperparameters
    hparams = None
    if params_class is not None:
        params_path = (HPARAMS_DIR / alg_name / (model_name.replace("/", "_") + ".json"))
        hparams = params_class.from_json(params_path)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

    # Load data
    with open(os.path.join("data", ds_name + ".json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    suffix = ("_" + "_".join(sorted(edit_formats)) if edit_formats else "")
    suffix = "" if suffix == "_" else suffix
    if memit_merge:
        suffix = f"_merge" + suffix
        if single_key or single_value:
            suffix += f"_{single_key}_{single_value}"
    output_path = "results/evaluation/{}/{}_{}{}.jsonl".format(model_name.replace("/", "_"), ds_name, alg_name, suffix)
    # output_path = "results/evaluation/TIME_TRACKING_{}/{}_{}{}.jsonl".format(model_name.replace("/", "_"), ds_name, alg_name, suffix)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="Evaluating"):
            case_id = item["case_id"]

            # Compute weight changes + record weights that changed
            etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

            target_edits = []
            if "completion" in edit_formats:
                target_edits.append({"case_id": case_id, **item["requested_rewrite"]})
            if "triplet" in edit_formats and "requested_rewrite_triplet" in item:
                target_edits.append({"case_id": f"{case_id}_triplet", **item["requested_rewrite_triplet"]})
            if "ODQA" in edit_formats and "requested_rewrite_odqa" in item:
                target_edits.append({"case_id": f"{case_id}_ODQA", **item["requested_rewrite_odqa"]})
            if "MC" in edit_formats and "requested_rewrite_MC" in item:
                target_edits.append({"case_id": f"{case_id}_MC", **item["requested_rewrite_MC"]})
            if "TF" in edit_formats and "requested_rewrite_TF" in item:
                target_edits.append({"case_id": f"{case_id}_TF", **item["requested_rewrite_TF"]})
            if "YN" in edit_formats and "requested_rewrite_YN" in item:
                target_edits.append({"case_id": f"{case_id}_YN", **item["requested_rewrite_YN"]})

            if alg_name in ["MEMIT"] and not target_edits:
                print(f"Warning: No target edits for case {case_id} with formats {edit_formats}. Skipping.")
                continue
            
            start_time = time.time()
            edited_model, weights_copy = apply_algo(
                model,
                tokenizer,
                target_edits,
                hparams,
                copy=False,
                return_orig_weights=True,
                memit_merge=True if memit_merge else False,
                single_key=single_key,
                single_value=single_value,
                **etc_args,
            )
            execution_time = time.time() - start_time
            debug = False
            if debug:
                print("DEBUG(outer):", execution_time)

            # Evaluation
            result = {"case_id": case_id}

            for fmt, prefix in [
                ("requested_rewrite", ""),
                ("requested_rewrite_triplet", "triplet_"),
                ("requested_rewrite_odqa", "ODQA_"),
                ("requested_rewrite_MC", "MC_"),
                ("requested_rewrite_TF", "TF_"),
                ("requested_rewrite_YN", "YN_")
            ]:
                if fmt not in item:
                    continue
                
                # Efficacy
                score, delta, predicted_token = compute_efficacy(item[fmt], tokenizer, edited_model, device)
                result[f"{prefix}efficacy_score"] = score
                result[f"{prefix}efficacy_magnitude"] = round(delta, 5)
                result[f"{prefix}predicted_token"] = predicted_token

                # Specificity
                score, delta = compute_specificity(item[fmt], tokenizer, None, edited_model, device)
                result[f"{prefix}specificity_score"] = score
                result[f"{prefix}specificity_magnitude"] = round(delta, 5)

            if "paraphrase_prompts" in item:
                paraphrase_prompts = item["paraphrase_prompts"]
            elif "rephrase_prompt" in item:
                paraphrase_prompts = item["rephrase_prompt"]
            elif "rephrase" in item:
                paraphrase_prompts = item["rephrase"]
            else:
                paraphrase_prompts = []
            
            if type(paraphrase_prompts) == str:
                paraphrase_prompts = [paraphrase_prompts]

            for idx, prompt in enumerate(paraphrase_prompts):
                prefix = f"paraphrase_{idx}_"

                # Efficacy with paraphrases
                new_item = deepcopy(item["requested_rewrite"])
                new_item["prompt"] = prompt
                
                score, delta, predicted_token = compute_efficacy(new_item, tokenizer, edited_model, device)
                result[f"{prefix}efficacy_score"] = score
                result[f"{prefix}efficacy_magnitude"] = round(delta, 5)
                result[f"{prefix}predicted_token"] = predicted_token

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Restore original weights
            if weights_copy is not None:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(model, k)[...] = v.to("cuda")

    print(f"\nSaved per-case evaluation results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline efficacy per case (format-specific)")
    parser.add_argument("--alg_name", type=str, required=True, help="Name of the knowledge editing algorithm")
    parser.add_argument("--model", type=str, required=True, help="Model name or local path")
    parser.add_argument("--ds_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--use_cache", dest="use_cache", action="store_true", help="Use cached k/v pairs")
    parser.add_argument("--edit_formats", nargs="+", type=str, default=["completion"], choices=["", "completion", "triplet", "ODQA", "MC", "TF", "YN"], help="List of formats to edit")
    parser.add_argument("--memit_merge", dest="memit_merge", action="store_true", help="Use memit_merge")
    parser.add_argument("--single_key", dest="single_key", action="store_true", help="Use single key")
    parser.add_argument("--single_value", dest="single_value", action="store_true", help="Use single value")
    args = parser.parse_args()

    evaluate_all_per_case(args.ds_name, args.model, args.alg_name, args.use_cache, args.edit_formats, args.memit_merge, args.single_key, args.single_value)
