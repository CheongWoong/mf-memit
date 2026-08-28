import json


def load_relation_templates(template_path):
    relation_map = {}
    with open(template_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            relation_id = item["relation"]
            relation_map[relation_id] = {
                "completion": item["template"],
                "qa": item["QA_template"],
                "triplet": item["triplet_template"],
                "yn": item["yn_template"]
            }
    return relation_map

def add_multiformat_to_dataset(counterfact_path, relation_template_path, output_path):
    with open(counterfact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    relation_templates = load_relation_templates(relation_template_path)
    updated_completion, updated_triplet, updated_ODQA, updated_MC, updated_TF, updated_YN = [], [], [], [], [], []

    for example in data:
        original = example.get("requested_rewrite", {})
        relation_id = original.get("relation_id")
        subject = original.get("subject")
        target_new = original.get("target_new", {}).get("str")
        target_true = original.get("target_true", {}).get("str")

        if not all([relation_id, subject, target_new, target_true]):
            continue
        if relation_id not in relation_templates:
            continue

        templates = relation_templates[relation_id]
        completion_template = templates["completion"]
        qa_template = templates["qa"]
        triplet_template = templates["triplet"]
        yn_template = templates["yn"]

        # Format 1: Original (completion)
        completion_prompt = completion_template.format(subject)

        example_completion = {
            "known_id": example["case_id"],
            "subject": subject,
            "attribute": target_true,
            "template": completion_template,
            "prompt": completion_prompt,
            "relation_id": relation_id,
        }

        # Format 2: Triplet
        triplet_prompt = triplet_template.format(subject)

        example_triplet = {
            "known_id": example["case_id"],
            "subject": subject,
            "attribute": target_true,
            "template": triplet_template,
            "prompt": triplet_prompt,
            "relation_id": relation_id,
        }

        # Format 3: ODQA
        odqa_template = (
            f"Question: {qa_template}\n"
            "Answer:"
        )
        odqa_prompt = odqa_template.format(subject)

        example_ODQA = {
            "known_id": example["case_id"],
            "subject": subject,
            "attribute": target_true,
            "template": odqa_template,
            "prompt": odqa_prompt,
            "relation_id": relation_id,
        }

        # Format 4: MCQA
        if example["case_id"] % 2 == 0:
            MC_template = (
                "Output only answer letter.\n"
                f"Question: {qa_template}\n"
                f"Options: A. {target_true} B. {target_new}\n"
                "Answer:"
            )
            MC_prompt = MC_template.format(subject)

            example_MC = {
                "known_id": example["case_id"],
                "subject": subject,
                "attribute": "A",
                "template": MC_template,
                "prompt": MC_prompt,
                "relation_id": relation_id,
            }
        else:
            MC_template = (
                "Output only answer letter.\n"
                f"Question: {qa_template}\n"
                f"Options: A. {target_new} B. {target_true}\n"
                "Answer:"
            )
            MC_prompt = MC_template.format(subject)

            example_MC = {
                "known_id": example["case_id"],
                "subject": subject,
                "attribute": "B",
                "template": MC_template,
                "prompt": MC_prompt,
                "relation_id": relation_id,
            }

        # Format 5: True/False
        if example["case_id"] % 2 == 0:
            TF_template = (
                "True or False?\n"
                f"Statement: {completion_template} {target_true}\n"
                "Answer:"
            )
            TF_prompt = TF_template.format(subject)

            example_TF = {
                "known_id": example["case_id"],
                "subject": subject,
                "attribute": "True",
                "template": TF_template,
                "prompt": TF_prompt,
                "relation_id": relation_id,
            }
        else:
            TF_template = (
                "True or False?\n"
                f"Statement: {completion_template} {target_new}\n"
                "Answer:"
            )
            TF_prompt = TF_template.format(subject)

            example_TF = {
                "known_id": example["case_id"],
                "subject": subject,
                "attribute": "False",
                "template": TF_template,
                "prompt": TF_prompt,
                "relation_id": relation_id,
            }

        # Format 6: YN
        if example["case_id"] % 2 == 0:
            YN_template = (
                "Yes or No?\n"
                f"Question: {yn_template} {target_true}?\n"
                "Answer:"
            )
            YN_prompt = YN_template.format(subject)

            example_YN = {
                "known_id": example["case_id"],
                "subject": subject,
                "attribute": "Yes",
                "template": YN_template,
                "prompt": YN_prompt,
                "relation_id": relation_id,
            }
        else:
            YN_template = (
                "Yes or No?\n"
                f"Question: {yn_template} {target_new}?\n"
                "Answer:"
            )
            YN_prompt = YN_template.format(subject)

            example_YN = {
                "known_id": example["case_id"],
                "subject": subject,
                "attribute": "No",
                "template": YN_template,
                "prompt": YN_prompt,
                "relation_id": relation_id,
            }

        updated_completion.append(example_completion)
        updated_triplet.append(example_triplet)
        updated_ODQA.append(example_ODQA)
        updated_MC.append(example_MC)
        updated_TF.append(example_TF)
        updated_YN.append(example_YN)

        if len(updated_completion) == 1000:
            with open(output_path.replace(".json", "_completion.json"), "w", encoding="utf-8") as f_out:
                json.dump(updated_completion, f_out, ensure_ascii=False, indent=2)
            with open(output_path.replace(".json", "_triplet.json"), "w", encoding="utf-8") as f_out:
                json.dump(updated_triplet, f_out, ensure_ascii=False, indent=2)
            with open(output_path.replace(".json", "_ODQA.json"), "w", encoding="utf-8") as f_out:
                json.dump(updated_ODQA, f_out, ensure_ascii=False, indent=2)
            with open(output_path.replace(".json", "_MC.json"), "w", encoding="utf-8") as f_out:
                json.dump(updated_MC, f_out, ensure_ascii=False, indent=2)
            with open(output_path.replace(".json", "_TF.json"), "w", encoding="utf-8") as f_out:
                json.dump(updated_TF, f_out, ensure_ascii=False, indent=2)
            with open(output_path.replace(".json", "_YN.json"), "w", encoding="utf-8") as f_out:
                json.dump(updated_YN, f_out, ensure_ascii=False, indent=2)

    print("Multiformat counterfact for causal tracing saved.")


if __name__ == "__main__":
    add_multiformat_to_dataset(
        counterfact_path="data/multi_counterfact.json",
        relation_template_path="data/relations.jsonl",
        output_path="data/causal_tracing.json"
    )