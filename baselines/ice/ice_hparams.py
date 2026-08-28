from dataclasses import dataclass
from typing import Optional, List
from util.hparams import HyperParams


@dataclass
class ICEHyperParams(HyperParams):
    # Template selection: "note", "contrast", or "instruct"
    template_type: str = "note"
    
    # Hint position configuration
    hint_position: str = "prefix"    # ["prefix", "suffix"]
    separator: str = "\n"            # Separator between hint and original prompt
    
    def get_template(self, template_type: str = None) -> dict:
        """
        Get template configuration based on type.
        Returns template info with format string and description.
        """
        if template_type is None:
            template_type = self.template_type

        fewshot_instruction = "You are given updated knowledge in the form of (subject, relation, object). Always answer queries consistently with this updated knowledge.\n\n"

        fewshot_update_no_retain = (
            "---\nUpdated knowledge: (Subair, occupation, composer)\n"
            "The occupation of Subair is composer.\n"
            "---\nUpdated knowledge: (Umayyad Caliphate, capital, Athens)\n"
            "The capital of Umayyad Caliphate is Athens.\n"
            "---\nUpdated knowledge: (Bong Jung-keun, position played on team, outfielder)\n"
            "Bong Jung-keun plays in the position of outfielder.\n"
            "---\n"
        )

        # with 2 retain examples
        fewshot_update = (
            "---\nUpdated knowledge: (Subair, occupation, composer)\n"
            "The occupation of Subair is composer.\n"
            "---\nUpdated knowledge: (Jean-Pierre Dionnet, languages spoken, written or signed, Spanish)\n"
            "Georges Pompidou used to communicate in French.\n"
            "---\nUpdated knowledge: (Umayyad Caliphate, capital, Athens)\n"
            "The capital of Umayyad Caliphate is Athens.\n"
            "---\nUpdated knowledge: (Georges Bernier, native language, Russian)\n"
            "The native language of Jean-Luc Picard is French.\n"
            "---\nUpdated knowledge: (Bong Jung-keun, position played on team, outfielder)\n"
            "Bong Jung-keun plays in the position of outfielder.\n"
            "---\n"
        )
        # K=2: completion, TF
        fewshot_update_k2 = (
            "---\nUpdated knowledge: (Subair, occupation, composer)\n"
            "The occupation of Subair is composer.\n"
            "---\nUpdated knowledge: (Jean-Pierre Dionnet, languages spoken, written or signed, Spanish)\n"
            "Georges Pompidou used to communicate in French.\n"
            "---\nUpdated knowledge: (Umayyad Caliphate, capital, Athens)\n"
            "True or False?\n"
            "Statement: The capital of Umayyad Caliphate is Athens.\n"
            "Answer: True\n"
            "---\nUpdated knowledge: (Georges Bernier, native language, Russian)\n"
            "True or False?\n"
            "Statement: The native language of Jean-Luc Picard is Russian.\n"
            "Answer: False\n"
            "---\nUpdated knowledge: (Bong Jung-keun, position played on team, outfielder)\n"
            "Bong Jung-keun plays in the position of outfielder.\n"
            "---\n"
        )
        fewshot_update_k3 = ""

        templates = {
            "note": {
                "name": "note",
                "format": "Note: {prompt_filled}\n\nNow:",
                "description": "Simple note format"
            },
            "contrast": {
                "name": "contrast", 
                "format": "Incorrect claim: {prompt_true_filled}\nCorrect claim: {prompt_new_filled}\nAlways answer using the correct claim.\n\nNow:",
                "description": "Correction format with explicit incorrect/correct claims"
            },
            "instruct": {
                "name": "instruct",
                "format": "Rule: {prompt_new_filled}\nAny questions related to this must follow this proposition. This applies whether the question is open-ended, True/False, or multiple-choice.\n\nNow:",
                "description": "Rule-based format with explicit instruction for all question types"
            },

            "naive": {
                "name": "naive",
                "format": "{prompt_filled}",
                "description": "Naive"
            },
            "instruction": {
                "name": "instruction",
                "format": "Assume that {prompt_new_filled}\nBased on the assumption, answer the following:",
                "description": "Explicit instruction"
            },
            "demonstration": {
                "name": "demonstration",
                "format": fewshot_instruction + fewshot_update + "\nNow answer the following query based on the updated knowledge:\n\nUpdated knowledge: {updated_knowledge}\n",
                "description": "Demonstration guided"
            },
            "demonstration_no_retain": {
                "name": "demonstration_no_retain",
                "format": fewshot_instruction + fewshot_update_no_retain + "\nNow answer the following query based on the updated knowledge:\n\nUpdated knowledge: {updated_knowledge}\n",
                "description": "Demonstration guided (without retain)"
            },
            "demonstration_K2": {
                "name": "demonstration_K2",
                "format": fewshot_instruction + fewshot_update_k2 + "\nNow answer the following query based on the updated knowledge:\n\nUpdated knowledge: {updated_knowledge}\n",
                "description": "Demonstration guided"
            },
        }
        
        return templates.get(template_type, templates["note"])
    
    def fill_prompt_template(self, prompt: str, subject: str, target: str) -> str:
        """
        Fill prompt template by replacing {} with subject and adding target.
        Example: "The native language of {} is" -> "The native language of Danielle Darrieux is English"
        """
        filled_prompt = prompt.format(subject)
        return filled_prompt + " " + target + "."

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = f.read()
        return cls(**eval(data))