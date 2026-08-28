import os
import json
from copy import deepcopy
import torch
import argparse
import hashlib
import re
import warnings
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from util import nethook
from util.globals import *


import sys
sys.path.append('../EasyEdit')
from easyeditor import BaseEditor
from easyeditor import (
    AlphaEditHyperParams,
    FTHyperParams,
    GraceHyperParams,
    LoRAHyperParams,
    MEMITHyperParams,
    PMETHyperParams,
    ROMEHyperParams,
    WISEHyperParams,
)

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
        outputs = model(input_ids=full_input_ids)
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
            outputs = model(input_ids=enc.input_ids)
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
                original_outputs = original_model(input_ids=original_enc.input_ids)
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


def has_literal_format_brace(prompt):
    return "{" in prompt or "}" in prompt


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
        prompt = "{} " + prompt

    new_item["prompt"] = ensure_space_around_braces(prompt)
    new_item["target_new"] = {"str": record["targets"][0]}
    new_item["target_true"] = {"str": ""}
    new_item["neighborhood_prompts"] = []
    return new_item


def _materialize_prompt(target_edit):
    prompt = target_edit["prompt"]
    if "{}" in prompt:
        return prompt.format(target_edit["subject"])
    return prompt


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _module_path_from_inner_param(inner_param):
    module_path = inner_param
    if module_path.endswith(".weight"):
        module_path = module_path[: -len(".weight")]
    module_path = module_path.replace("[", ".").replace("]", "")
    return module_path


def get_direction_module_name(alg_name, hparams, result_alg_name=None):
    """Return the module whose last-prompt-token output defines the edit direction."""
    output_alg_name = result_alg_name or alg_name
    if alg_name == "LoRA" and output_alg_name.startswith("LoRA-local-down"):
        layers = getattr(hparams, "layers", None) or [7]
        modules = getattr(hparams, "target_modules", None) or ["down_proj"]
        if isinstance(modules, str):
            modules = [modules]
        if len(layers) != 1 or len(modules) != 1:
            return None
        return f"model.layers.{layers[0]}.mlp.{modules[0]}"
    if alg_name in {"WISE", "GRACE"}:
        inner_params = getattr(hparams, "inner_params", None) or []
        if not inner_params:
            return None
        return _module_path_from_inner_param(inner_params[0])
    return None


def unwrap_easyedit_model(model_like):
    """WISE/GRACE wrap the HF model; PEFT LoRA may wrap the base model."""
    if hasattr(model_like, "model") and not hasattr(model_like, "config") and not hasattr(model_like, "generate"):
        return model_like.model
    return model_like


def resolve_trace_layer(model, module_name):
    candidates = [
        module_name,
        module_name[len("model.") :] if module_name.startswith("model.") else module_name,
        f"base_model.model.{module_name}",
        f"base_model.{module_name}",
        f"model.{module_name}",
    ]
    for candidate in candidates:
        try:
            nethook.get_module(model, candidate)
            return candidate
        except LookupError:
            continue
        except AttributeError:
            continue
    raise LookupError(f"Could not find module {module_name!r} in {type(model).__name__}")


def capture_last_prompt_representation(model, tokenizer, module_name, prompt, device):
    """Capture output representation at the source prompt's last non-padding token."""
    raw_model = unwrap_easyedit_model(model)
    trace_layer = resolve_trace_layer(raw_model, module_name)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        with nethook.Trace(
            module=raw_model,
            layer=trace_layer,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = raw_model(**enc)
    output = tr.output[0] if isinstance(tr.output, tuple) else tr.output
    token_index = int(enc["attention_mask"][0].sum().item()) - 1
    return output[0, token_index].detach().float().cpu().numpy(), token_index, trace_layer


def save_representation_edit_direction(
    direction_dir,
    model_name,
    ds_name,
    method_name,
    case_id,
    source_format,
    module_name,
    trace_layer,
    token_index,
    prompt,
    target_new,
    before,
    after,
):
    model_dir = model_name.replace("/", "_")
    out_dir = os.path.join(direction_dir, model_dir, ds_name, method_name, source_format)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"case_{_safe_name(case_id)}.npz")
    direction = after - before
    np.savez_compressed(
        path,
        direction=direction,
        before=before,
        after=after,
        metadata=np.array(json.dumps({
            "case_id": case_id,
            "method": method_name,
            "source_format": source_format,
            "module_name": module_name,
            "trace_layer": trace_layer,
            "token_index": token_index,
            "prompt": prompt,
            "target_new": target_new,
            "direction_kind": "post_minus_pre_source_last_token_representation",
        }, ensure_ascii=False)),
    )
    return path


def is_lora_local_direction(alg_name, result_alg_name):
    output_alg_name = result_alg_name or alg_name
    return alg_name == "LoRA" and output_alg_name.startswith("LoRA-local-down")


def _get_lora_adapter_names(module):
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return []
    lora_a = module.lora_A
    if hasattr(lora_a, "keys"):
        return list(lora_a.keys())
    return ["default"]


def _get_lora_weight(container, adapter_name):
    value = container[adapter_name] if hasattr(container, "__getitem__") else container
    if hasattr(value, "weight"):
        value = value.weight
    return value.detach().float().cpu().numpy()


def extract_lora_adapter_factors(model):
    """Return LoRA delta-W factors without materializing the full dense update."""
    factors = []
    for module_name, module in model.named_modules():
        for adapter_name in _get_lora_adapter_names(module):
            scaling = getattr(module, "scaling", 1.0)
            if isinstance(scaling, dict):
                scaling = scaling.get(adapter_name, next(iter(scaling.values())))
            factors.append({
                "module_name": module_name,
                "adapter_name": adapter_name,
                "A": _get_lora_weight(module.lora_A, adapter_name),
                "B": _get_lora_weight(module.lora_B, adapter_name),
                "scaling": float(scaling),
            })
    return factors


def save_lora_adapter_edit_direction(
    direction_dir,
    model_name,
    ds_name,
    method_name,
    case_id,
    source_format,
    edit_sources,
    edited_model,
):
    factors = extract_lora_adapter_factors(edited_model)
    if not factors:
        raise RuntimeError(f"No LoRA adapter factors found for case {case_id}.")
    if len(factors) != 1:
        raise RuntimeError(
            f"Expected one LoRA adapter module for {method_name}, found {len(factors)}: "
            f"{[factor['module_name'] for factor in factors]}"
        )

    factor = factors[0]
    model_dir = model_name.replace("/", "_")
    out_dir = os.path.join(direction_dir, model_dir, ds_name, method_name, source_format)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"case_{_safe_name(case_id)}.npz")
    np.savez_compressed(
        path,
        A=factor["A"],
        B=factor["B"],
        scaling=np.array(factor["scaling"], dtype=np.float32),
        metadata=np.array(json.dumps({
            "case_id": case_id,
            "method": method_name,
            "source_format": source_format,
            "module_name": factor["module_name"],
            "adapter_name": factor["adapter_name"],
            "edit_sources": edit_sources,
            "direction_kind": "lora_delta_weight_factors",
            "delta_weight_formula": "delta_W = scaling * B @ A",
        }, ensure_ascii=False)),
    )
    return path


def infer_source_format_from_edit(target_edit, fallback):
    case_id = str(target_edit.get("case_id", ""))
    if "_paraphrase_" in case_id:
        return "paraphrase"
    if "_portability_" in case_id:
        return "portability"
    if case_id.endswith("_triplet"):
        return "triplet"
    if case_id.endswith("_ODQA"):
        return "ODQA"
    if case_id.endswith("_MC"):
        return "MC"
    if case_id.endswith("_MC1"):
        return "MC1"
    if case_id.endswith("_MC2"):
        return "MC2"
    if case_id.endswith("_TF"):
        return "TF"
    if case_id.endswith("_YN"):
        return "YN"
    if fallback == "multi":
        return "completion"
    return fallback


def restore_easyedit_model(editor, alg_name, edited_model, weights_copy, device):
    """Restore per-case edits for EasyEdit methods with method-specific state."""
    if alg_name in {"LoRA", "QLoRA"}:
        if hasattr(edited_model, "unload"):
            base_model = edited_model.unload()
            if base_model is not None:
                editor.model = base_model
        if hasattr(editor.model, "peft_config"):
            del editor.model.peft_config
        torch.cuda.empty_cache()
        return

    if alg_name in {"GRACE", "WISE"} and callable(weights_copy):
        with torch.no_grad():
            weights_copy()
        torch.cuda.empty_cache()
        return

    if weights_copy is not None:
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(editor.model, k)[...] = v.to(device)


def evaluate_all_per_case(
    ds_name,
    model_name,
    alg_name,
    use_cache,
    edit_formats=None,
    memit_merge=False,
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
    lora_num_steps_override=None,
    lora_layers_override=None,
    lora_target_modules_override=None,
    result_alg_name=None,
    resume=False,
    save_edit_directions=False,
    direction_dir="results/edit_directions/single_format_transfer",
    max_cases=None,
):
    """
    Evaluates each case (question) across all 3 formats, and writes jsonl output.
    """
    if edit_formats is None:
        edit_formats = []

    # Set algorithm-specific variables
    if alg_name == "ROME":
        hparams = ROMEHyperParams.from_hparams(f"./hparams/EasyEdit/ROME/{model_name.split('/')[-1]}.yaml")
    elif alg_name == "PMET":
        hparams = PMETHyperParams.from_hparams(f"./hparams/EasyEdit/PMET/{model_name.split('/')[-1]}.yaml")
    elif alg_name == "AlphaEdit":
        hparams = AlphaEditHyperParams.from_hparams(f"./hparams/EasyEdit/AlphaEdit/{model_name.split('/')[-1]}.yaml")
    elif alg_name == "MEMIT2":
        hparams = MEMITHyperParams.from_hparams(f"./hparams/EasyEdit/MEMIT/{model_name.split('/')[-1]}.yaml")
    elif alg_name == "LoRA":
        hparams = LoRAHyperParams.from_hparams(f"./hparams/EasyEdit/LoRA/{model_name.split('/')[-1]}.yaml")
        if lora_num_steps_override is not None:
            print(f"Overriding LoRA num_steps: {hparams.num_steps} -> {lora_num_steps_override}")
            hparams.num_steps = lora_num_steps_override
        if lora_layers_override is not None:
            old_layers = hparams.layers
            if lora_layers_override.lower() in {"", "all", "none", "[]"}:
                hparams.layers = []
            else:
                hparams.layers = [int(layer) for layer in lora_layers_override.split(",") if layer]
            print(f"Overriding LoRA layers: {old_layers} -> {hparams.layers}")
        if lora_target_modules_override is not None:
            old_modules = hparams.target_modules
            hparams.target_modules = lora_target_modules_override
            print(f"Overriding LoRA target_modules: {old_modules} -> {hparams.target_modules}")
    elif alg_name == "WISE":
        hparams = WISEHyperParams.from_hparams(f"./hparams/EasyEdit/WISE/{model_name.split('/')[-1]}.yaml")
    elif alg_name == "FT":
        hparams = FTHyperParams.from_hparams(f"./hparams/EasyEdit/FT/{model_name.split('/')[-1]}.yaml")
    elif alg_name == "GRACE":
        hparams = GraceHyperParams.from_hparams(f"./hparams/EasyEdit/GRACE/{model_name.split('/')[-1]}.yaml")
    else:
        raise NotImplementedError

    if layers_override is not None:
        if not hasattr(hparams, "layers"):
            raise ValueError(f"{alg_name} hparams does not expose a layers field.")
        old_layers = hparams.layers
        hparams.layers = [int(layer) for layer in layers_override.split(",") if layer.strip()]
        print(f"Overriding {alg_name} layers: {old_layers} -> {hparams.layers}")

    # Load model
    editor = BaseEditor.from_hparams(hparams)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

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
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    suffix = ("_" + "_".join(sorted(edit_formats)) if edit_formats else "")
    suffix = "" if suffix == "_" else suffix
    if memit_merge:
        suffix = f"_merge" + suffix
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
    output_alg_name = result_alg_name or alg_name
    output_path = "results/evaluation/{}/{}_{}{}.jsonl".format(model_name.replace("/", "_"), ds_name, output_alg_name, suffix)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    completed_case_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                try:
                    completed_case_ids.add(json.loads(line)["case_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming from {output_path}; skipping {len(completed_case_ids)} completed case_id(s).")

    output_mode = "a" if resume else "w"
    with open(output_path, output_mode, encoding="utf-8") as out_f:
        for item in tqdm(data, desc="Evaluating"):
            case_id = item["case_id"]
            if case_id in completed_case_ids:
                continue

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
                    if has_literal_format_brace(prompt):
                        print(
                            f"Warning: skipping paraphrase edit with literal brace "
                            f"for case {case_id}: {prompt[:120]!r}"
                        )
                        continue
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

            if not target_edits:
                print(f"Warning: No target edits for case {case_id} with formats {edit_formats}. Skipping.")
                continue
            
            
            # assert len(target_edits) == 1
            # target_edit = target_edits[0]
            direction_captures = []
            lora_direction_payload = None
            if save_edit_directions:
                output_alg_name_for_direction = result_alg_name or alg_name
                if is_lora_local_direction(alg_name, result_alg_name):
                    fallback_format = edit_formats[0] if len(edit_formats) == 1 else "multi"
                    edit_sources = []
                    for target_edit in target_edits:
                        edit_sources.append({
                            "source_format": infer_source_format_from_edit(target_edit, fallback_format),
                            "prompt": _materialize_prompt(target_edit),
                            "target_new": target_edit["target_new"]["str"],
                        })
                    lora_direction_payload = {
                        "method_name": output_alg_name_for_direction,
                        "source_format": edit_sources[0]["source_format"] if len(edit_sources) == 1 else "merged",
                        "edit_sources": edit_sources,
                    }
                else:
                    module_name = get_direction_module_name(alg_name, hparams, result_alg_name)
                    if module_name is None:
                        warnings.warn(
                            f"Edit direction saving is not configured for {output_alg_name_for_direction}; "
                            f"case {case_id} is evaluated without a direction artifact."
                        )
                    else:
                        direction_tokenizer = getattr(editor, "tok", tokenizer)
                        fallback_format = edit_formats[0] if len(edit_formats) == 1 else "multi"
                        for target_edit in target_edits:
                            source_format = infer_source_format_from_edit(target_edit, fallback_format)
                            source_prompt = _materialize_prompt(target_edit)
                            before_vec, token_index, before_trace_layer = capture_last_prompt_representation(
                                editor.model,
                                direction_tokenizer,
                                module_name,
                                source_prompt,
                                device,
                            )
                            direction_captures.append({
                                "before": before_vec,
                                "token_index": token_index,
                                "before_trace_layer": before_trace_layer,
                                "module_name": module_name,
                                "source_format": source_format,
                                "source_prompt": source_prompt,
                                "target_new": target_edit["target_new"]["str"],
                                "method_name": output_alg_name_for_direction,
                            })

            edit_kwargs = {}
            if alg_name == "WISE":
                edit_kwargs["loc_prompts"] = [
                    target_edit["subject"] for target_edit in target_edits
                ]

            _, edited_model, weights_copy = editor.edit(
                prompts=[_materialize_prompt(target_edit) for target_edit in target_edits],
                ground_truth=None,
                target_new=[target_edit['target_new']['str'] for target_edit in target_edits],
                subject=[target_edit['subject'] for target_edit in target_edits],
                sequential_edit=True,
                **edit_kwargs,
            )
            edited_model.eval()

            if lora_direction_payload is not None:
                save_lora_adapter_edit_direction(
                    direction_dir=direction_dir,
                    model_name=model_name,
                    ds_name=ds_name,
                    method_name=lora_direction_payload["method_name"],
                    case_id=case_id,
                    source_format=lora_direction_payload["source_format"],
                    edit_sources=lora_direction_payload["edit_sources"],
                    edited_model=edited_model,
                )

            for direction_capture in direction_captures:
                after_vec, after_token_index, after_trace_layer = capture_last_prompt_representation(
                    edited_model,
                    direction_tokenizer,
                    direction_capture["module_name"],
                    direction_capture["source_prompt"],
                    device,
                )
                if after_token_index != direction_capture["token_index"]:
                    warnings.warn(
                        f"Direction token index changed for case {case_id}: "
                        f"{direction_capture['token_index']} -> {after_token_index}"
                    )
                save_representation_edit_direction(
                    direction_dir=direction_dir,
                    model_name=model_name,
                    ds_name=ds_name,
                    method_name=direction_capture["method_name"],
                    case_id=case_id,
                    source_format=direction_capture["source_format"],
                    module_name=direction_capture["module_name"],
                    trace_layer=after_trace_layer,
                    token_index=after_token_index,
                    prompt=direction_capture["source_prompt"],
                    target_new=direction_capture["target_new"],
                    before=direction_capture["before"],
                    after=after_vec,
                )

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

            # Restore original weights/state for the next independent edit case.
            restore_easyedit_model(editor, alg_name, edited_model, weights_copy, device)

    print(f"\nSaved per-case evaluation results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline efficacy per case (format-specific)")
    parser.add_argument("--alg_name", type=str, required=True, help="Name of the knowledge editing algorithm")
    parser.add_argument("--model", type=str, required=True, help="Model name or local path")
    parser.add_argument("--ds_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--use_cache", dest="use_cache", action="store_true", help="Use cached k/v pairs")
    parser.add_argument("--edit_formats", nargs="+", type=str, default=["completion"], choices=["", "completion", "triplet", "ODQA", "MC", "MC1", "MC2", "TF", "YN"], help="List of formats to edit")
    parser.add_argument("--memit_merge", dest="memit_merge", action="store_true", help="Use memit_merge")
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
    parser.add_argument("--layers_override", type=str, default=None, help="Override hparams.layers for layer ablation runs")
    parser.add_argument("--lora_num_steps_override", type=int, default=None, help="Override LoRA num_steps for smoke/ablation runs")
    parser.add_argument("--lora_layers_override", type=str, default=None, help="Override LoRA layers; use 'all' for all layers or comma-separated layer ids")
    parser.add_argument("--lora_target_modules_override", nargs="+", type=str, default=None, help="Override LoRA target modules")
    parser.add_argument("--result_alg_name", type=str, default=None, help="Optional algorithm name used only in the output filename")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output file and skip completed case_id rows")
    parser.add_argument("--save_edit_directions", action="store_true", help="Save per-case source-format edit direction vectors")
    parser.add_argument("--direction_dir", type=str, default="results/edit_directions/single_format_transfer", help="Directory for saved edit direction npz files")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of cases for smoke tests")
    args = parser.parse_args()

    evaluate_all_per_case(
        args.ds_name,
        args.model,
        args.alg_name,
        args.use_cache,
        args.edit_formats,
        args.memit_merge,
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
        args.lora_num_steps_override,
        args.lora_layers_override,
        args.lora_target_modules_override,
        args.result_alg_name,
        args.resume,
        args.save_edit_directions,
        args.direction_dir,
        args.max_cases,
    )
