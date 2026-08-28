"""Evaluation helpers shared by AlphaEdit sequential-editing experiments."""

from itertools import combinations

import numpy as np

from .evaluate_baselines_with_easyedit import compute_efficacy, compute_specificity


EVAL_FORMATS = [
    ("requested_rewrite", "completion"),
    ("requested_rewrite_triplet", "triplet"),
    ("requested_rewrite_odqa", "ODQA"),
    ("requested_rewrite_MC", "MC"),
    ("requested_rewrite_TF", "TF"),
    ("requested_rewrite_YN", "YN"),
]
EDIT_FIELDS = {
    "completion": "requested_rewrite",
    "triplet": "requested_rewrite_triplet",
    "ODQA": "requested_rewrite_odqa",
    "MC": "requested_rewrite_MC",
    "TF": "requested_rewrite_TF",
    "YN": "requested_rewrite_YN",
}


def materialize_request(item, field, case_id, suffix):
    request = {"case_id": f"{case_id}{suffix}", **item[field]}
    if "{}" in request["prompt"]:
        request["prompt"] = request["prompt"].format(request["subject"])
    return request


def build_fact_requests(item, case_id, edit_formats):
    requests = []
    for edit_format in edit_formats:
        field = EDIT_FIELDS[edit_format]
        if field not in item:
            raise KeyError(f"Case {case_id} has no {field} request.")
        suffix = "" if edit_format == "completion" else f"_{edit_format}"
        requests.append(materialize_request(item, field, case_id, suffix))
    return requests


def pairwise_consistency(scores):
    pairs = list(combinations(scores, 2))
    return float(np.mean([left == right for left, right in pairs]))


def evaluate_cases(data, model, tokenizer, device):
    rows = []
    for item in data:
        format_scores = {}
        row = {"case_id": item["case_id"]}
        for field, name in EVAL_FORMATS:
            score, magnitude, predicted_token = compute_efficacy(
                item[field], tokenizer, model, device
            )
            format_scores[name] = int(score)
            row[f"{name}_efficacy_score"] = int(score)
            row[f"{name}_efficacy_magnitude"] = round(float(magnitude), 5)
            row[f"{name}_predicted_token"] = predicted_token

        specificity, specificity_magnitude = compute_specificity(
            item["requested_rewrite"], tokenizer, None, model, device
        )
        row["completion_specificity_score"] = float(specificity)
        row["completion_specificity_magnitude"] = round(
            float(specificity_magnitude), 5
        )
        row["cross_format_generalization"] = float(
            np.mean(
                [format_scores[name] for name in ["triplet", "ODQA", "MC", "TF", "YN"]]
            )
        )
        row["cross_format_consistency"] = pairwise_consistency(
            list(format_scores.values())
        )
        row["consistent_accuracy"] = int(all(format_scores.values()))
        rows.append(row)
    return rows


def summarize(rows):
    keys = [
        "completion_efficacy_score",
        "cross_format_generalization",
        "cross_format_consistency",
        "consistent_accuracy",
        "completion_specificity_score",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
