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

def add_multiformat_to_dataset(counterfact_path, relation_template_path, output_path, factual_variant=False):
    with open(counterfact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    relation_templates = load_relation_templates(relation_template_path)
    updated = []

    for example in data:
        if factual_variant:
            example["requested_rewrite"]["target_new"], example["requested_rewrite"]["target_true"] = example["requested_rewrite"]["target_true"], example["requested_rewrite"]["target_new"]

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
        neighborhood_prompts = example.get("neighborhood_prompts", [])

        # Format 1: Original (completion)
        completion_prompt = completion_template

        example["requested_rewrite"] = {
            "prompt": completion_prompt,
            "relation_id": relation_id,
            "target_new": {"str": target_new},
            "target_true": {"str": target_true},
            "subject": subject,
            "neighborhood_prompts": neighborhood_prompts
        }

        # Format 2: Triplet
        triplet_prompt = triplet_template

        example["requested_rewrite_triplet"] = {
            "prompt": triplet_prompt,
            "relation_id": relation_id,
            "target_new": {"str": target_new},
            "target_true": {"str": target_true},
            "subject": subject,
            "neighborhood_prompts": []
        }

        # Format 3: ODQA
        odqa_prompt = (
            f"Question: {qa_template}\n"
            "Answer:"
        )

        odqa_neighborhood_prompts = [
            f"Question: {p}\nAnswer:"
            for p in neighborhood_prompts
        ]

        example["requested_rewrite_odqa"] = {
            "prompt": odqa_prompt,
            "relation_id": relation_id,
            "target_new": {"str": target_new},
            "target_true": {"str": target_true},
            "subject": subject,
            "neighborhood_prompts": odqa_neighborhood_prompts
        }

        # Format 4: MCQA
        if example["case_id"] % 2 == 0:
            mcqa_prompt = (
                "Output only answer letter.\n"
                f"Question: {qa_template}\n"
                f"Options: A. {target_new} B. {target_true}\n"
                "Answer:"
            )
            
            mc_neighborhood_prompts = [
                f"Output only answer letter.\nQuestion: {p}\nOptions: A. {target_new} B. {target_true}\nAnswer:"
                for p in neighborhood_prompts
            ]

            example["requested_rewrite_MC"] = {
                "prompt": mcqa_prompt,
                "relation_id": relation_id,
                "target_new": {"str": "A"},
                "target_true": {"str": "B"},
                "subject": subject,
                "neighborhood_prompts": mc_neighborhood_prompts
            }
        else:
            mcqa_prompt = (
                "Output only answer letter.\n"
                f"Question: {qa_template}\n"
                f"Options: A. {target_true} B. {target_new}\n"
                "Answer:"
            )

            mc_neighborhood_prompts = [
                f"Output only answer letter.\nQuestion: {p}\nOptions: A. {target_true} B. {target_new}\nAnswer:"
                for p in neighborhood_prompts
            ]

            example["requested_rewrite_MC"] = {
                "prompt": mcqa_prompt,
                "relation_id": relation_id,
                "target_new": {"str": "B"},
                "target_true": {"str": "A"},
                "subject": subject,
                "neighborhood_prompts": mc_neighborhood_prompts
            }

        # Format 5: True/False
        tf_prompt = (
            "True or False?\n"
            f"Statement: {completion_template} {target_new}\n"
            "Answer:"
        )
        
        tf_neighborhood_prompts = [
            f"True or False?\nStatement: {p} {target_new}\nAnswer:"
            for p in neighborhood_prompts
        ]

        example["requested_rewrite_TF"] = {
            "prompt": tf_prompt,
            "relation_id": relation_id,
            "target_new": {"str": "True"},
            "target_true": {"str": "False"},
            "subject": subject,
            "neighborhood_prompts": tf_neighborhood_prompts
        }

        # Format 6: YN
        yn_prompt = (
            "Yes or No?\n"
            f"Question: {yn_template} {target_new}?\n"
            "Answer:"
        )
        
        yn_neighborhood_prompts = [
            f"Yes or No?\nQuestion: {p} {target_new}\nAnswer:"
            for p in neighborhood_prompts
        ]

        example["requested_rewrite_YN"] = {
            "prompt": yn_prompt,
            "relation_id": relation_id,
            "target_new": {"str": "Yes"},
            "target_true": {"str": "No"},
            "subject": subject,
            "neighborhood_prompts": yn_neighborhood_prompts
        }

        # example.pop("paraphrase_prompts")
        example.pop("neighborhood_prompts")
        example.pop("attribute_prompts")
        example.pop("generation_prompts")

        updated.append(example)

        if len(updated) == 1000:
            with open(output_path.replace(".json", "_1000.json"), "w", encoding="utf-8") as f_out:
                json.dump(updated, f_out, ensure_ascii=False, indent=2)        

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(updated, f_out, ensure_ascii=False, indent=2)

    if factual_variant:
        print("Multiformat factual variant saved.")
    else:
        print("Multiformat counterfact saved.")


if __name__ == "__main__":
    add_multiformat_to_dataset(
        counterfact_path="data/multi_counterfact.json",
        relation_template_path="data/relations.jsonl",
        output_path="data/multiformat_counterfact.json"
    )
    add_multiformat_to_dataset(
        counterfact_path="data/multi_counterfact.json",
        relation_template_path="data/relations.jsonl",
        output_path="data/multiformat_fact.json",
        factual_variant=True
    )