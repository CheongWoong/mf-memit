import argparse
import json
from pathlib import Path


SUBSET_CONFIG = {
    "zsre": {
        "input": "data/knowedit/benchmark/ZsRE/ZsRE-test-all.json",
        "output": "data/knowedit/benchmark/ZsRE/multiformat_ZsRE_test.json",
        "include_mc": True,
    },
    "wiki_recent": {
        "input": "data/knowedit/benchmark/wiki_recent/recent_test.json",
        "output": "data/knowedit/benchmark/wiki_recent/multiformat_recent_test.json",
        "include_mc": False,
    },
    "wiki_counterfact": {
        "input": "data/knowedit/benchmark/wiki_counterfact/test_cf.json",
        "output": "data/knowedit/benchmark/wiki_counterfact/multiformat_test_cf.json",
        "include_mc": True,
    },
}

KEEP_KEYS = {
    # "subject",
    # "prompt",
    # "target_new",
    # "ground_truth",
    "rephrase",
    "rephrase_prompt",
    "portability"
}


def make_prompt_template(prompt: str, subject: str) -> str:
    if "{}" in prompt:
        return prompt
    if subject and subject in prompt:
        return prompt.replace(subject, "{}", 1)
    return "{} " + prompt


def clean_prompt_for_statement(prompt_template: str) -> str:
    prompt = prompt_template.rstrip()
    if prompt.endswith("?"):
        prompt = prompt[:-1]
    return prompt.rstrip()


def pick_target_true(item):
    target_true = item.get("ground_truth")
    if isinstance(target_true, list):
        return target_true[0] if target_true else ""
    if isinstance(target_true, str):
        return target_true
    return ""


def build_multiformat_entry(item, case_id, include_mc):
    subject = item.get("subject", "")
    prompt_raw = item.get("prompt", "")
    prompt_template = make_prompt_template(prompt_raw, subject)

    target_new = item.get("target_new", "")
    target_true = pick_target_true(item)
    target_true_missing = target_true == ""

    statement_base = clean_prompt_for_statement(prompt_template)

    example = {"case_id": case_id}
    for key in KEEP_KEYS:
        if key in item:
            example[key] = item[key]
    if target_true_missing:
        example["target_true_missing"] = True

    example["requested_rewrite"] = {
        "prompt": prompt_template,
        "relation_id": item.get("relation_id", ""),
        "target_new": {"str": target_new},
        "target_true": {"str": target_true},
        "subject": subject,
        "neighborhood_prompts": [],
    }

    example["requested_rewrite_odqa"] = {
        "prompt": f"Question: {prompt_template}\nAnswer:",
        "relation_id": item.get("relation_id", ""),
        "target_new": {"str": target_new},
        "target_true": {"str": target_true},
        "subject": subject,
        "neighborhood_prompts": [],
    }

    if include_mc and target_true != "":
        if case_id % 2 == 0:
            mcqa_prompt = (
                "Output only answer letter.\n"
                f"Question: {prompt_template}\n"
                f"Options: A. {target_new} B. {target_true}\n"
                "Answer:"
            )
            example["requested_rewrite_MC"] = {
                "prompt": mcqa_prompt,
                "relation_id": item.get("relation_id", ""),
                "target_new": {"str": "A"},
                "target_true": {"str": "B"},
                "subject": subject,
                "neighborhood_prompts": [],
            }
        else:
            mcqa_prompt = (
                "Output only answer letter.\n"
                f"Question: {prompt_template}\n"
                f"Options: A. {target_true} B. {target_new}\n"
                "Answer:"
            )
            example["requested_rewrite_MC"] = {
                "prompt": mcqa_prompt,
                "relation_id": item.get("relation_id", ""),
                "target_new": {"str": "B"},
                "target_true": {"str": "A"},
                "subject": subject,
                "neighborhood_prompts": [],
            }

    example["requested_rewrite_TF"] = {
        "prompt": (
            "True or False?\n"
            f"Statement: {statement_base} {target_new}\n"
            "Answer:"
        ),
        "relation_id": item.get("relation_id", ""),
        "target_new": {"str": "True"},
        "target_true": {"str": "False"},
        "subject": subject,
        "neighborhood_prompts": [],
    }

    example["requested_rewrite_YN"] = {
        "prompt": (
            "Yes or No?\n"
            f"Question: {statement_base} {target_new}?\n"
            "Answer:"
        ),
        "relation_id": item.get("relation_id", ""),
        "target_new": {"str": "Yes"},
        "target_true": {"str": "No"},
        "subject": subject,
        "neighborhood_prompts": [],
    }

    return example


def generate_subset(subset_name: str):
    cfg = SUBSET_CONFIG[subset_name]
    input_path = Path(cfg["input"])
    output_path = Path(cfg["output"])

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated = []
    for idx, item in enumerate(data):
        updated.append(build_multiformat_entry(item, idx, cfg["include_mc"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        json.dump(updated, f_out, ensure_ascii=False, indent=2)

    print(f"{subset_name}: saved {len(updated)} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all"] + sorted(SUBSET_CONFIG.keys()),
        help="Which KnowEdit subset to process.",
    )
    args = parser.parse_args()

    if args.subset == "all":
        for name in SUBSET_CONFIG:
            generate_subset(name)
    else:
        generate_subset(args.subset)


if __name__ == "__main__":
    main()
