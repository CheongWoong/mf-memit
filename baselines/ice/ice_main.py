from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ice_hparams import ICEHyperParams


import jsonlines
triplet_templates = {}
with jsonlines.open("data/relations.jsonl") as fin:
    for line in fin.iter():
        triplet_templates[line["relation"]] = line["triplet_template"].replace("( subject, relation, object )\n( ", "(")

def apply_ice_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ICEHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Dummy function for ICE compatibility with other editing methods.
    ICE doesn't modify model weights - actual prompt modification happens during evaluation.
    
    :param copy: If true, will preserve the original model while creating a new one.
    :param return_orig_weights: For compatibility with other methods.
    :return: (1) the model (unchanged), (2) empty weights dict
    """
    
    if copy:
        model = deepcopy(model)
    
    # Store ICE metadata for evaluation phase
    model._ice_hparams = hparams
    model._ice_requests = requests  # Store current requests for evaluation
    
    print(f"ICE ready for evaluation with {len(requests)} requests using template: {hparams.template_type}")
    
    # Return empty weights dict for compatibility
    weights_copy = {}
    return model, weights_copy


def generate_hint(prompt: str, relation_id, subject: str, target_new: str, target_true: str, hparams: ICEHyperParams, template_type=None) -> str:
    """
    Generate hint text based on the selected template.
    """
    template_info = hparams.get_template(template_type)
    template_format = template_info["format"]
    
    # Fill prompt templates for different targets
    prompt_new_filled = hparams.fill_prompt_template(prompt, subject, target_new)
    prompt_true_filled = hparams.fill_prompt_template(prompt, subject, target_true)
    if relation_id in triplet_templates:
        updated_knowledge = triplet_templates[relation_id].format(subject) + " " + target_new + ")"
    else:
        updated_knowledge = f"({subject}, {relation_id}," + " " + target_new + ")"
    
    try:
        hint = template_format.format(
            prompt_filled=prompt_new_filled,
            prompt_new_filled=prompt_new_filled,
            prompt_true_filled=prompt_true_filled,
            updated_knowledge=updated_knowledge
        )
    except KeyError as e:
        # Fallback to simple note template if formatting fails
        hint = f"Note: {prompt_new_filled}"
        print(f"Warning: Template formatting failed, using fallback. Error: {e}")
    
    return hint


def modify_prompt_with_ice(prompt: str, entry: Dict, model: AutoModelForCausalLM, template_type=None) -> str:
    """
    Modify prompt with ICE hint based on current evaluation entry.
    This function is called during evaluation.
    The hint is always based on the original completion format, regardless of the current prompt format.
    """
    if not hasattr(model, '_ice_hparams') or not hasattr(model, '_ice_requests'):
        return prompt
    
    hparams = model._ice_hparams
    
    # Always use the base completion format to generate consistent hints
    # Find the original request that matches this entry's subject
    original_request = None
    for req in model._ice_requests:
        if req["subject"] == entry["subject"]:
            original_request = req
            break
    
    if not original_request:
        return prompt  # No matching request found
    
    # Generate hint based on the original completion format request
    hint = generate_hint(
        original_request["prompt"], 
        original_request["relation_id"], 
        original_request["subject"], 
        original_request["target_new"]["str"], 
        original_request["target_true"]["str"], 
        hparams,
        template_type
    )
    
    if hparams.hint_position == "prefix":
        modified_prompt = hint + hparams.separator + prompt
    else:  # suffix
        modified_prompt = prompt + hparams.separator + hint
    
    return modified_prompt