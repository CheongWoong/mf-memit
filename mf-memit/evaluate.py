import time
import os
from copy import deepcopy
import json
import torch
import argparse
import hashlib
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

EVAL_FORMAT_SPECS = [
    ("requested_rewrite", ""),
    ("requested_rewrite_triplet", "triplet_"),
    ("requested_rewrite_odqa", "ODQA_"),
    ("requested_rewrite_MC", "MC_"),
    ("requested_rewrite_MC1", "MC1_"),
    ("requested_rewrite_MC2", "MC2_"),
    ("requested_rewrite_TF", "TF_"),
    ("requested_rewrite_YN", "YN_"),
]


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


def compute_portability(prompt, targets, tokenizer, model, device):
    """
    Computes teacher-forcing token accuracy for portability targets.
    Returns the best score across target aliases.
    """
    if isinstance(targets, str):
        targets = [targets]
    targets = [str(t) for t in targets if str(t).strip()]
    if not targets:
        return -1

    scores = []
    before_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for target in targets:
            prompt_target = prompt.rstrip() + " " + target.lstrip()
            max_prompt_len = len(tokenizer.encode(prompt_target)) + 1
            prompt_target_tok = tokenizer(
                [prompt_target],
                padding=True,
                truncation=True,
                max_length=max(1024, max_prompt_len),
                return_tensors="pt",
            ).to(device)
            prompt_tok = tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                max_length=max(1024, max_prompt_len),
                return_tensors="pt",
            )

            num_prompt_toks = int((prompt_tok["input_ids"][0] != tokenizer.pad_token_id).sum())
            num_pad_toks = int((prompt_target_tok["input_ids"][0].cpu() == tokenizer.pad_token_id).sum())
            prompt_len = num_pad_toks + num_prompt_toks

            with torch.no_grad():
                outputs = model(**prompt_target_tok)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                answers = torch.argmax(logits, dim=-1)[0].detach().cpu().numpy().tolist()
                labels = prompt_target_tok["input_ids"][0].detach().cpu().numpy().tolist()

            answer_tokens = answers[prompt_len - 1 : -1]
            label_tokens = labels[prompt_len:]
            if not label_tokens:
                continue
            scores.append(float(sum(a == b for a, b in zip(answer_tokens, label_tokens))) / len(label_tokens))
    finally:
        tokenizer.padding_side = before_padding_side

    return max(scores) if scores else -1


def ensure_space_around_braces(prompt: str) -> str:
    token = "{}"
    i = prompt.find(token)
    if i < 1 or i + 2 >= len(prompt):
        return prompt
    left = " " if not prompt[i - 1].isspace() else ""
    right = " " if not prompt[i + 2].isspace() else ""
    return prompt[:i] + left + token + right + prompt[i + 2:]


def get_paraphrase_prompts(item):
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

    return paraphrase_prompts


def flatten_ground_truth(ground_truth):
    if ground_truth is None:
        return []
    if isinstance(ground_truth, str):
        return [ground_truth]
    if not isinstance(ground_truth, list):
        return [str(ground_truth)]

    targets = []
    for item in ground_truth:
        if isinstance(item, list):
            targets.extend([str(x) for x in item])
        else:
            targets.append(str(item))
    return [x for x in targets if x.strip()]


def get_portability_records(item, allowed_categories=None, require_subject_count=None):
    records = []
    global_idx = 0
    portability = item.get("portability", {})
    if not isinstance(portability, dict):
        return records
    subject = item.get("requested_rewrite", {}).get("subject", "")

    for category, prompts in portability.items():
        if not isinstance(prompts, list):
            prompts = [prompts]
        for category_idx, prompt_gt in enumerate(prompts):
            if not isinstance(prompt_gt, dict):
                continue
            prompt = prompt_gt.get("prompt", "")
            targets = flatten_ground_truth(prompt_gt.get("ground_truth", []))
            if not prompt or not targets:
                global_idx += 1
                continue
            subject_count = prompt.count(subject) if subject else 0
            if allowed_categories is not None and category not in allowed_categories:
                global_idx += 1
                continue
            if require_subject_count is not None and subject_count != require_subject_count:
                global_idx += 1
                continue
            records.append({
                "global_idx": global_idx,
                "category": category,
                "category_idx": category_idx,
                "prompt": prompt,
                "targets": targets,
                "subject_count": subject_count,
            })
            global_idx += 1
    return records


def get_relation_specificity_records(item):
    records = []
    locality = item.get("locality", {})
    if not isinstance(locality, dict):
        return records

    prompts = locality.get("Relation_Specificity", [])
    if not isinstance(prompts, list):
        prompts = [prompts]

    for category_idx, prompt_gt in enumerate(prompts):
        if not isinstance(prompt_gt, dict):
            continue
        prompt = prompt_gt.get("prompt", "")
        targets = flatten_ground_truth(prompt_gt.get("ground_truth", []))
        if not prompt or not targets:
            continue
        records.append({
            "category": "Relation_Specificity",
            "category_idx": category_idx,
            "prompt": prompt,
            "targets": targets,
        })
    return records


def split_portability_records(
    item,
    fold2_ratio=0.5,
    fold1_max=None,
    allowed_categories=None,
    require_subject_count=None,
):
    """
    Deterministically splits portability prompts within each case.
    If a case has only one portability prompt, it is assigned only to fold1.
    """
    records = get_portability_records(
        item,
        allowed_categories=allowed_categories,
        require_subject_count=require_subject_count,
    )
    if len(records) <= 1:
        return {"Portability_fold1": records, "Portability_fold2": []}

    case_id = item.get("case_id", "")

    def stable_key(record):
        raw = f"{case_id}|{record['category']}|{record['category_idx']}|{record['prompt']}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    ordered = sorted(records, key=stable_key)
    if fold1_max is not None:
        fold1_count = min(int(fold1_max), len(ordered) - 1)
        fold1_count = max(1, fold1_count)
        fold1 = ordered[:fold1_count]
        fold2 = ordered[fold1_count:]
    else:
        fold2_count = max(1, int(round(len(ordered) * fold2_ratio)))
        fold2 = ordered[:fold2_count]
        fold1 = ordered[fold2_count:]
        if not fold1:
            fold1, fold2 = ordered[:-1], ordered[-1:]
    return {"Portability_fold1": fold1, "Portability_fold2": fold2}


def build_paraphrase_edit(item, prompt):
    new_item = deepcopy(item["requested_rewrite"])
    subject = new_item["subject"]

    # MEMIT expects a prompt template with "{}" marking the subject location.
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    if subject in prompt:
        prompt = prompt.replace(subject, "{}", 1)
    new_item["prompt"] = ensure_space_around_braces(prompt)
    return new_item


def build_portability_edit(item, record):
    new_item = deepcopy(item["requested_rewrite"])
    subject = new_item["subject"]
    prompt = record["prompt"].replace("{", "{{").replace("}", "}}")
    if subject and subject in prompt:
        prompt = prompt.replace(subject, "{}", 1)
    elif "{}" not in prompt:
        # MEMIT/ROME locate the edit at a subject token. If the portability
        # prompt uses an alias or omits the exact subject, keep the original
        # prompt intact and prepend the canonical subject as the lookup anchor.
        prompt = "{} " + prompt

    new_item["prompt"] = ensure_space_around_braces(prompt)
    new_item["target_new"] = {"str": record["targets"][0]}
    new_item["target_true"] = {"str": ""}
    new_item["neighborhood_prompts"] = []
    return new_item


def build_target_edits_for_item(
    item,
    edit_formats,
    edit_paraphrases=False,
    max_edit_paraphrases=None,
    edit_paraphrase_subject_count=None,
    edit_portability=False,
    portability_edit_fold="Portability_fold1",
    portability_fold2_ratio=0.5,
    portability_fold1_max=None,
    portability_edit_categories=None,
    portability_edit_subject_count=None,
    max_edit_portability=None,
):
    case_id = item["case_id"]
    target_edits = []
    if "completion" in edit_formats:
        target_edits.append({"case_id": case_id, **item["requested_rewrite"]})
    if edit_paraphrases:
        paraphrase_prompts_for_edit = get_paraphrase_prompts(item)
        if edit_paraphrase_subject_count is not None:
            subject = item["requested_rewrite"]["subject"]
            paraphrase_prompts_for_edit = [
                prompt
                for prompt in paraphrase_prompts_for_edit
                if prompt.count(subject) == edit_paraphrase_subject_count
            ]
        if max_edit_paraphrases is not None:
            paraphrase_prompts_for_edit = paraphrase_prompts_for_edit[:max_edit_paraphrases]
        for idx, prompt in enumerate(paraphrase_prompts_for_edit):
            target_edits.append({
                "case_id": f"{case_id}_paraphrase_{idx}",
                **build_paraphrase_edit(item, prompt),
            })
    if edit_portability:
        portability_splits = split_portability_records(
            item,
            fold2_ratio=portability_fold2_ratio,
            fold1_max=portability_fold1_max,
            allowed_categories=set(portability_edit_categories) if portability_edit_categories else None,
            require_subject_count=portability_edit_subject_count,
        )
        portability_records_for_edit = list(portability_splits.get(portability_edit_fold, []))
        if portability_edit_fold == "all":
            portability_records_for_edit = get_portability_records(
                item,
                allowed_categories=set(portability_edit_categories) if portability_edit_categories else None,
                require_subject_count=portability_edit_subject_count,
            )
        if max_edit_portability is not None:
            portability_records_for_edit = portability_records_for_edit[:max_edit_portability]
        for record in portability_records_for_edit:
            target_edits.append({
                "case_id": f"{case_id}_portability_{record['global_idx']}",
                **build_portability_edit(item, record),
            })
    if "triplet" in edit_formats and "requested_rewrite_triplet" in item:
        target_edits.append({"case_id": f"{case_id}_triplet", **item["requested_rewrite_triplet"]})
    if "ODQA" in edit_formats and "requested_rewrite_odqa" in item:
        target_edits.append({"case_id": f"{case_id}_ODQA", **item["requested_rewrite_odqa"]})
    if "MC" in edit_formats and "requested_rewrite_MC" in item:
        target_edits.append({"case_id": f"{case_id}_MC", **item["requested_rewrite_MC"]})
    if "MC1" in edit_formats and "requested_rewrite_MC1" in item:
        target_edits.append({"case_id": f"{case_id}_MC1", **item["requested_rewrite_MC1"]})
    if "MC2" in edit_formats and "requested_rewrite_MC2" in item:
        target_edits.append({"case_id": f"{case_id}_MC2", **item["requested_rewrite_MC2"]})
    if "TF" in edit_formats and "requested_rewrite_TF" in item:
        target_edits.append({"case_id": f"{case_id}_TF", **item["requested_rewrite_TF"]})
    if "YN" in edit_formats and "requested_rewrite_YN" in item:
        target_edits.append({"case_id": f"{case_id}_YN", **item["requested_rewrite_YN"]})
    return target_edits


def evaluate_item(item, tokenizer, edited_model, device, eval_portability=False, eval_relation_specificity=False, portability_fold2_ratio=0.5):
    result = {"case_id": item["case_id"]}

    for fmt, prefix in EVAL_FORMAT_SPECS:
        if fmt not in item:
            continue
        score, delta, predicted_token = compute_efficacy(item[fmt], tokenizer, edited_model, device)
        result[f"{prefix}efficacy_score"] = score
        result[f"{prefix}efficacy_magnitude"] = round(delta, 5)
        result[f"{prefix}predicted_token"] = predicted_token

        score, delta = compute_specificity(item[fmt], tokenizer, None, edited_model, device)
        result[f"{prefix}specificity_score"] = score
        result[f"{prefix}specificity_magnitude"] = round(delta, 5)

    for idx, prompt in enumerate(get_paraphrase_prompts(item)):
        prefix = f"paraphrase_{idx}_"
        new_item = deepcopy(item["requested_rewrite"])
        new_item["prompt"] = prompt
        score, delta, predicted_token = compute_efficacy(new_item, tokenizer, edited_model, device)
        result[f"{prefix}efficacy_score"] = score
        result[f"{prefix}efficacy_magnitude"] = round(delta, 5)
        result[f"{prefix}predicted_token"] = predicted_token

    if eval_portability:
        portability_splits = split_portability_records(item, fold2_ratio=portability_fold2_ratio)
        split_by_global_idx = {}
        for fold_name, split_records in portability_splits.items():
            for record in split_records:
                split_by_global_idx[record["global_idx"]] = fold_name
        for record in get_portability_records(item):
            prompt = record["prompt"]
            if "{}" in prompt:
                prompt = prompt.format(item["requested_rewrite"]["subject"])
            score = compute_portability(prompt, record["targets"], tokenizer, edited_model, device)
            fold_name = split_by_global_idx.get(record["global_idx"], "Portability_unknown")
            prefix = f"{record['category']}_{record['global_idx']}_"
            result[f"{prefix}portability_score"] = round(score, 5)
            result[f"{prefix}portability_fold"] = fold_name
            result[f"{prefix}portability_category_idx"] = record["category_idx"]

    if eval_relation_specificity:
        for record in get_relation_specificity_records(item):
            prompt = record["prompt"]
            if "{}" in prompt:
                prompt = prompt.format(item["requested_rewrite"]["subject"])
            score = compute_portability(prompt, record["targets"], tokenizer, edited_model, device)
            prefix = f"Relation_Specificity_{record['category_idx']}_"
            result[f"{prefix}score"] = round(score, 5)

    return result


def evaluate_all_per_case(
    ds_name,
    model_name,
    alg_name,
    use_cache,
    edit_formats=None,
    memit_merge=False,
    single_key=False,
    single_value=False,
    edit_paraphrases=False,
    max_edit_paraphrases=None,
    edit_paraphrase_subject_count=None,
    eval_portability=False,
    eval_relation_specificity=False,
    edit_portability=False,
    portability_edit_fold="Portability_fold1",
    portability_fold2_ratio=0.5,
    portability_fold1_max=None,
    portability_edit_categories=None,
    portability_edit_subject_count=None,
    max_edit_portability=None,
    layers_override=None,
    num_edits=None,
    mom2_update_weight_override=None,
    kl_factor_override=None,
    v_weight_decay_override=None,
    max_cases=None,
):
    """
    Evaluates each case (question) across all 3 formats, and writes jsonl output.
    """
    if edit_formats is None:
        edit_formats = []
    # Set algorithm-specific variables
    base_alg_name = alg_name
    if alg_name.startswith("MEMIT-"):
        base_alg_name = "MEMIT"

    params_class, apply_algo = ALG_DICT.get(base_alg_name, (None, lambda *x, **kwargs: (x[0], None)))

    # Get run hyperparameters
    hparams = None
    if params_class is not None:
        params_path = (HPARAMS_DIR / base_alg_name / (model_name.replace("/", "_") + ".json"))
        hparams = params_class.from_json(params_path)
        if layers_override is not None:
            hparams.layers = [int(layer) for layer in layers_override.split(",") if layer.strip()]
        if mom2_update_weight_override is not None:
            old_weight = hparams.mom2_update_weight
            hparams.mom2_update_weight = float(mom2_update_weight_override)
            print(f"Overriding mom2_update_weight: {old_weight} -> {hparams.mom2_update_weight}")
        if kl_factor_override is not None:
            old_factor = hparams.kl_factor
            hparams.kl_factor = float(kl_factor_override)
            print(f"Overriding kl_factor: {old_factor} -> {hparams.kl_factor}")
        if v_weight_decay_override is not None:
            old_decay = hparams.v_weight_decay
            hparams.v_weight_decay = float(v_weight_decay_override)
            print(f"Overriding v_weight_decay: {old_decay} -> {hparams.v_weight_decay}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

    # Load data
    with open(os.path.join("data", ds_name + ".json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_cases is not None:
        data = data[:max_cases]

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{base_alg_name}"
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
    if layers_override is not None:
        suffix += f"_layers{layers_override.replace(',', '-')}"
    if num_edits is not None:
        suffix += f"_batch{num_edits}"
        if mom2_update_weight_override is not None:
            suffix += f"_lambda{str(mom2_update_weight_override).replace('.', 'p')}"
    if kl_factor_override is not None:
        suffix += f"_kl{str(kl_factor_override).replace('.', 'p')}"
    if v_weight_decay_override is not None:
        suffix += f"_vwd{str(v_weight_decay_override).replace('.', 'p')}"
    if max_cases is not None:
        suffix += f"_maxcases{max_cases}"
    if edit_paraphrases:
        suffix += "_with_paraphrase_edits"
    if edit_portability:
        suffix += f"_with_{portability_edit_fold}_edits"
        if portability_edit_categories:
            suffix += "_cats_" + "-".join(portability_edit_categories)
        if portability_edit_subject_count is not None:
            suffix += f"_subject{portability_edit_subject_count}"
    if eval_portability:
        suffix += "_with_portability_eval"
    if eval_relation_specificity:
        suffix += "_with_relation_specificity_eval"
    output_path = "results/evaluation/{}/{}_{}{}.jsonl".format(model_name.replace("/", "_"), ds_name, alg_name, suffix)
    # output_path = "results/evaluation/TIME_TRACKING_{}/{}_{}{}.jsonl".format(model_name.replace("/", "_"), ds_name, alg_name, suffix)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if num_edits is not None:
        if base_alg_name != "MEMIT":
            raise ValueError("--num_edits batch mode is currently implemented for MEMIT/MEMIT-XF only.")
        if num_edits <= 0:
            raise ValueError("--num_edits must be positive.")
        with open(output_path, "w", encoding="utf-8") as out_f:
            for batch_start in tqdm(range(0, len(data), num_edits), desc="Evaluating batches"):
                batch_items = data[batch_start : batch_start + num_edits]
                batch_target_edits = []
                group_sizes = []
                for item in batch_items:
                    edits = build_target_edits_for_item(
                        item,
                        edit_formats,
                        edit_paraphrases,
                        max_edit_paraphrases,
                        edit_paraphrase_subject_count,
                        edit_portability,
                        portability_edit_fold,
                        portability_fold2_ratio,
                        portability_fold1_max,
                        portability_edit_categories,
                        portability_edit_subject_count,
                        max_edit_portability,
                    )
                    if not edits:
                        print(f"Warning: No target edits for case {item['case_id']} with formats {edit_formats}. Skipping.")
                        continue
                    batch_target_edits.extend(edits)
                    group_sizes.append(len(edits))

                if not batch_target_edits:
                    continue

                etc_args = dict(cache_template=cache_template) if base_alg_name in ["ROME", "MEMIT"] else dict()
                if memit_merge:
                    etc_args["memit_merge_group_sizes"] = group_sizes

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                edit_start_time = time.perf_counter()
                edited_model, weights_copy = apply_algo(
                    model,
                    tokenizer,
                    batch_target_edits,
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    memit_merge=True if memit_merge else False,
                    single_key=single_key,
                    single_value=single_value,
                    **etc_args,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_edit_seconds = time.perf_counter() - edit_start_time
                batch_edit_seconds_per_case = batch_edit_seconds / len(batch_items)
                print(
                    f"Batch edit timing: {batch_edit_seconds:.4f}s total, "
                    f"{batch_edit_seconds_per_case:.6f}s/case "
                    f"({len(batch_items)} cases, {len(batch_target_edits)} requests)"
                )

                for item in batch_items:
                    result = evaluate_item(
                        item,
                        tokenizer,
                        edited_model,
                        device,
                        eval_portability=eval_portability,
                        eval_relation_specificity=eval_relation_specificity,
                        portability_fold2_ratio=portability_fold2_ratio,
                    )
                    result["batch_edit_seconds"] = batch_edit_seconds
                    result["batch_edit_seconds_per_case"] = batch_edit_seconds_per_case
                    result["batch_edit_num_cases"] = len(batch_items)
                    result["batch_edit_num_requests"] = len(batch_target_edits)
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

                if weights_copy is not None:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(model, k)[...] = v.to("cuda")

        print(f"\nSaved batched per-case evaluation results to: {output_path}")
        return

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="Evaluating"):
            case_id = item["case_id"]

            # Compute weight changes + record weights that changed
            etc_args = dict(cache_template=cache_template) if base_alg_name in ["ROME", "MEMIT"] else dict()

            target_edits = []
            if "completion" in edit_formats:
                target_edits.append({"case_id": case_id, **item["requested_rewrite"]})
            if edit_paraphrases:
                paraphrase_prompts_for_edit = get_paraphrase_prompts(item)
                if edit_paraphrase_subject_count is not None:
                    subject = item["requested_rewrite"]["subject"]
                    paraphrase_prompts_for_edit = [
                        prompt
                        for prompt in paraphrase_prompts_for_edit
                        if prompt.count(subject) == edit_paraphrase_subject_count
                    ]
                if max_edit_paraphrases is not None:
                    paraphrase_prompts_for_edit = paraphrase_prompts_for_edit[:max_edit_paraphrases]
                for idx, prompt in enumerate(paraphrase_prompts_for_edit):
                    target_edits.append({
                        "case_id": f"{case_id}_paraphrase_{idx}",
                        **build_paraphrase_edit(item, prompt),
                    })
            if edit_portability:
                portability_splits = split_portability_records(
                    item,
                    fold2_ratio=portability_fold2_ratio,
                    fold1_max=portability_fold1_max,
                    allowed_categories=set(portability_edit_categories) if portability_edit_categories else None,
                    require_subject_count=portability_edit_subject_count,
                )
                portability_records_for_edit = list(portability_splits.get(portability_edit_fold, []))
                if portability_edit_fold == "all":
                    portability_records_for_edit = get_portability_records(
                        item,
                        allowed_categories=set(portability_edit_categories) if portability_edit_categories else None,
                        require_subject_count=portability_edit_subject_count,
                    )
                if max_edit_portability is not None:
                    portability_records_for_edit = portability_records_for_edit[:max_edit_portability]
                for record in portability_records_for_edit:
                    target_edits.append({
                        "case_id": f"{case_id}_portability_{record['global_idx']}",
                        **build_portability_edit(item, record),
                    })
            if "triplet" in edit_formats and "requested_rewrite_triplet" in item:
                target_edits.append({"case_id": f"{case_id}_triplet", **item["requested_rewrite_triplet"]})
            if "ODQA" in edit_formats and "requested_rewrite_odqa" in item:
                target_edits.append({"case_id": f"{case_id}_ODQA", **item["requested_rewrite_odqa"]})
            if "MC" in edit_formats and "requested_rewrite_MC" in item:
                target_edits.append({"case_id": f"{case_id}_MC", **item["requested_rewrite_MC"]})
            if "MC1" in edit_formats and "requested_rewrite_MC1" in item:
                target_edits.append({"case_id": f"{case_id}_MC1", **item["requested_rewrite_MC1"]})
            if "MC2" in edit_formats and "requested_rewrite_MC2" in item:
                target_edits.append({"case_id": f"{case_id}_MC2", **item["requested_rewrite_MC2"]})
            if "TF" in edit_formats and "requested_rewrite_TF" in item:
                target_edits.append({"case_id": f"{case_id}_TF", **item["requested_rewrite_TF"]})
            if "YN" in edit_formats and "requested_rewrite_YN" in item:
                target_edits.append({"case_id": f"{case_id}_YN", **item["requested_rewrite_YN"]})

            if base_alg_name in ["MEMIT"] and not target_edits:
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

            for fmt, prefix in EVAL_FORMAT_SPECS:
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

            paraphrase_prompts = get_paraphrase_prompts(item)

            for idx, prompt in enumerate(paraphrase_prompts):
                prefix = f"paraphrase_{idx}_"

                # Efficacy with paraphrases
                new_item = deepcopy(item["requested_rewrite"])
                new_item["prompt"] = prompt
                
                score, delta, predicted_token = compute_efficacy(new_item, tokenizer, edited_model, device)
                result[f"{prefix}efficacy_score"] = score
                result[f"{prefix}efficacy_magnitude"] = round(delta, 5)
                result[f"{prefix}predicted_token"] = predicted_token

            if eval_portability:
                portability_splits = split_portability_records(item, fold2_ratio=portability_fold2_ratio)
                split_by_global_idx = {}
                for fold_name, split_records in portability_splits.items():
                    for record in split_records:
                        split_by_global_idx[record["global_idx"]] = fold_name

                for record in get_portability_records(item):
                    prompt = record["prompt"]
                    if "{}" in prompt:
                        prompt = prompt.format(item["requested_rewrite"]["subject"])
                    score = compute_portability(prompt, record["targets"], tokenizer, edited_model, device)
                    fold_name = split_by_global_idx.get(record["global_idx"], "Portability_unknown")
                    prefix = f"{record['category']}_{record['global_idx']}_"
                    result[f"{prefix}portability_score"] = round(score, 5)
                    result[f"{prefix}portability_fold"] = fold_name
                    result[f"{prefix}portability_category_idx"] = record["category_idx"]

            if eval_relation_specificity:
                for record in get_relation_specificity_records(item):
                    prompt = record["prompt"]
                    if "{}" in prompt:
                        prompt = prompt.format(item["requested_rewrite"]["subject"])
                    score = compute_portability(prompt, record["targets"], tokenizer, edited_model, device)
                    prefix = f"Relation_Specificity_{record['category_idx']}_"
                    result[f"{prefix}score"] = round(score, 5)

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
    parser.add_argument("--edit_formats", nargs="+", type=str, default=["completion"], choices=["", "completion", "triplet", "ODQA", "MC", "MC1", "MC2", "TF", "YN"], help="List of formats to edit")
    parser.add_argument("--memit_merge", dest="memit_merge", action="store_true", help="Use memit_merge")
    parser.add_argument("--single_key", dest="single_key", action="store_true", help="Use single key")
    parser.add_argument("--single_value", dest="single_value", action="store_true", help="Use single value")
    parser.add_argument("--edit_paraphrases", dest="edit_paraphrases", action="store_true", help="Add paraphrase prompts as edit requests")
    parser.add_argument("--max_edit_paraphrases", type=int, default=None, help="Maximum number of paraphrase prompts to add as edit requests")
    parser.add_argument("--edit_paraphrase_subject_count", type=int, default=None, help="Require this exact canonical-subject count in paraphrase edit prompts")
    parser.add_argument("--eval_portability", dest="eval_portability", action="store_true", help="Evaluate KnowEdit portability prompts when present")
    parser.add_argument("--eval_relation_specificity", dest="eval_relation_specificity", action="store_true", help="Evaluate KnowEdit locality.Relation_Specificity prompts when present")
    parser.add_argument("--edit_portability", dest="edit_portability", action="store_true", help="Add portability prompts as edit requests")
    parser.add_argument("--portability_edit_fold", type=str, default="Portability_fold1", choices=["Portability_fold1", "Portability_fold2", "all"], help="Which portability fold to use as edit requests")
    parser.add_argument("--portability_fold2_ratio", type=float, default=0.5, help="Per-case ratio assigned to Portability_fold2")
    parser.add_argument("--portability_fold1_max", type=int, default=None, help="Maximum number of portability prompts assigned to Portability_fold1 for edit-source splitting")
    parser.add_argument("--portability_edit_categories", nargs="+", type=str, default=None, help="Optional portability categories allowed as edit requests")
    parser.add_argument("--portability_edit_subject_count", type=int, default=None, help="Require this exact canonical-subject count in portability edit prompts")
    parser.add_argument("--max_edit_portability", type=int, default=None, help="Maximum number of portability prompts to add as edit requests per case")
    parser.add_argument("--layers_override", type=str, default=None, help="Override hparams.layers with comma-separated layer ids")
    parser.add_argument("--num_edits", type=int, default=None, help="Batch this many cases into one MEMIT update. Omit for independent single-edit evaluation.")
    parser.add_argument("--mom2_update_weight_override", type=float, default=None, help="Override MEMIT mom2_update_weight (lambda) for sensitivity runs.")
    parser.add_argument("--kl_factor_override", type=float, default=None, help="Override MEMIT kl_factor for sensitivity runs.")
    parser.add_argument("--v_weight_decay_override", type=float, default=None, help="Override MEMIT v_weight_decay for sensitivity runs.")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of cases for smoke tests.")
    args = parser.parse_args()

    evaluate_all_per_case(
        args.ds_name,
        args.model,
        args.alg_name,
        args.use_cache,
        args.edit_formats,
        args.memit_merge,
        args.single_key,
        args.single_value,
        args.edit_paraphrases,
        args.max_edit_paraphrases,
        args.edit_paraphrase_subject_count,
        args.eval_portability,
        args.eval_relation_specificity,
        args.edit_portability,
        args.portability_edit_fold,
        args.portability_fold2_ratio,
        args.portability_fold1_max,
        args.portability_edit_categories,
        args.portability_edit_subject_count,
        args.max_edit_portability,
        args.layers_override,
        args.num_edits,
        args.mom2_update_weight_override,
        args.kl_factor_override,
        args.v_weight_decay_override,
        args.max_cases,
    )
