from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch

from easyeditor.models.alphaedit.AlphaEdit_main import (
    get_context_templates,
    upd_matrix_match_shape,
)
from easyeditor.models.alphaedit.compute_ks import compute_ks
from easyeditor.models.alphaedit.compute_z import (
    compute_z,
    get_module_input_output_at_words,
)
from easyeditor.util import nethook


class SequentialAlphaEdit:
    """Apply joint AlphaEdit batches while retaining state across rounds."""

    def __init__(self, hparams, fact_chunk_size=10):
        self.hparams = hparams
        self.projection = torch.load(Path(hparams.P_loc), map_location="cpu")
        self.cache_c = None
        self.fact_chunk_size = fact_chunk_size
        self.context_templates = None

    @property
    def device(self):
        return torch.device(f"cuda:{self.hparams.device}")

    def reset_chain(self, model):
        last_layer = self.hparams.layers[-1]
        weight = nethook.get_parameter(
            model,
            f"{self.hparams.rewrite_module_tmp.format(last_layer)}.weight",
        )
        hidden_size = weight.shape[1]
        self.cache_c = torch.zeros(
            (len(self.hparams.layers), hidden_size, hidden_size),
            dtype=torch.float32,
            device="cpu",
        )

    def snapshot_weights(self, model):
        return {
            self._weight_name(layer): nethook.get_parameter(
                model, self._weight_name(layer)
            ).detach().cpu().clone()
            for layer in self.hparams.layers
        }

    def restore_weights(self, model, snapshot):
        with torch.no_grad():
            for name, value in snapshot.items():
                parameter = nethook.get_parameter(model, name)
                parameter[...] = value.to(parameter.device, dtype=parameter.dtype)

    def state_dict(self, model):
        if self.cache_c is None:
            raise RuntimeError("Sequential AlphaEdit chain has not been initialized.")
        return {
            "weights": self.snapshot_weights(model),
            "cache_c": self.cache_c.clone(),
            "context_templates": self.context_templates,
        }

    def load_state_dict(self, model, state_dict):
        self.restore_weights(model, state_dict["weights"])
        self.cache_c = state_dict["cache_c"].float().cpu().clone()
        self.context_templates = state_dict.get("context_templates")

    def apply_fact(self, model, tok, requests: List[Dict]):
        return self.apply_batch(model, tok, [requests])

    def apply_batch(self, model, tok, fact_requests: List[List[Dict]]):
        if self.cache_c is None:
            self.reset_chain(model)
        if not fact_requests or any(not requests for requests in fact_requests):
            raise ValueError("Sequential AlphaEdit requires requests for every fact.")

        prepared = [self._prepare_requests(requests) for requests in fact_requests]
        deltas = self._execute_batch(model, tok, prepared)

        with torch.no_grad():
            for name, update in deltas.items():
                weight = nethook.get_parameter(model, name)
                update = upd_matrix_match_shape(update.to(weight.device), weight.shape)
                weight[...] += update.to(weight.dtype)

        return model

    def _execute_batch(self, model, tok, fact_requests):
        hparams = self.hparams
        weights = {
            self._weight_name(layer): nethook.get_parameter(
                model, self._weight_name(layer)
            )
            for layer in hparams.layers
        }
        weights_before_batch = {
            name: weight.detach().clone() for name, weight in weights.items()
        }
        if self.context_templates is None:
            self.context_templates = get_context_templates(model, tok)
        context_templates = self.context_templates
        z_layer = hparams.layers[-1]

        # Each fact gets one value shared by its formats; facts are solved jointly.
        target_zs = [
            compute_z(
                model,
                tok,
                requests,
                hparams,
                z_layer,
                context_templates,
            ).detach().cpu()
            for requests in fact_requests
        ]

        deltas = {}
        for layer_idx, layer in enumerate(hparams.layers):
            projection = self.projection[layer_idx].to(self.device).float()
            covariance = self.cache_c[layer_idx].to(self.device)
            gram = torch.zeros_like(covariance)
            key_residual = torch.zeros(
                covariance.shape[0],
                target_zs[0].numel(),
                dtype=torch.float32,
                device=self.device,
            )

            for start in range(0, len(fact_requests), self.fact_chunk_size):
                request_groups = fact_requests[start : start + self.fact_chunk_size]
                chunk_targets = target_zs[start : start + self.fact_chunk_size]
                requests = [request for group in request_groups for request in group]
                layer_ks = compute_ks(
                    model, tok, requests, hparams, layer, context_templates
                ).T
                current_zs = get_module_input_output_at_words(
                    model,
                    tok,
                    z_layer,
                    context_templates=[request["prompt"] for request in requests],
                    words=[request["subject"] for request in requests],
                    module_template=hparams.layer_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                )[1].T
                target_per_request = torch.cat(
                    [
                        target.unsqueeze(1).expand(-1, len(group))
                        for target, group in zip(chunk_targets, request_groups)
                    ],
                    dim=1,
                ).to(self.device)
                targets = target_per_request - current_zs.to(self.device).float()
                repeat_factor = layer_ks.size(1) // targets.size(1)
                if repeat_factor < 1 or layer_ks.size(1) % targets.size(1) != 0:
                    raise ValueError(
                        f"Cannot align {layer_ks.size(1)} keys with "
                        f"{targets.size(1)} targets."
                    )
                residual = targets.repeat_interleave(repeat_factor, dim=1)
                residual /= len(hparams.layers) - layer_idx
                keys = layer_ks.to(self.device).float()
                gram += keys @ keys.T
                key_residual += keys @ residual.T
                del layer_ks, current_zs, target_per_request, targets, residual, keys
                torch.cuda.empty_cache()

            lhs = projection @ (gram + covariance)
            lhs += hparams.L2 * torch.eye(
                gram.shape[0], dtype=torch.float32, device=self.device
            )
            rhs = projection @ key_residual
            update = torch.linalg.solve(lhs, rhs)

            name = self._weight_name(layer)
            update = upd_matrix_match_shape(update, weights[name].shape)
            with torch.no_grad():
                weights[name][...] += update.to(weights[name].dtype)
            deltas[name] = update.detach().cpu()

            del covariance, gram, key_residual, lhs, rhs
            torch.cuda.empty_cache()

        # Preserve this batch's post-update keys for subsequent rounds.
        for layer_idx, layer in enumerate(hparams.layers):
            gram = torch.zeros_like(self.cache_c[layer_idx], device=self.device)
            for start in range(0, len(fact_requests), self.fact_chunk_size):
                request_groups = fact_requests[start : start + self.fact_chunk_size]
                requests = [request for group in request_groups for request in group]
                layer_ks = compute_ks(
                    model, tok, requests, hparams, layer, context_templates
                ).T.to(self.device).float()
                gram += layer_ks @ layer_ks.T
                del layer_ks
                torch.cuda.empty_cache()
            self.cache_c[layer_idx] += gram.cpu()
            del gram

        # The solve uses temporary layer-wise insertions; apply all deltas once below.
        with torch.no_grad():
            for name, weight in weights.items():
                weight[...] = weights_before_batch[name]

        return deltas

    def _prepare_requests(self, requests):
        prepared = deepcopy(requests)
        for request in prepared:
            target = request["target_new"]
            if isinstance(target, dict):
                target = target["str"]
            if not target.startswith(" "):
                target = " " + target
            request["target_new"] = target

            prompt = request["prompt"]
            subject = request["subject"]
            if "{}" not in prompt:
                if subject not in prompt:
                    raise ValueError(
                        f"Subject {subject!r} is absent from prompt {prompt!r}."
                    )
                prompt = prompt.replace(subject, "{}", 1)
            request["prompt"] = prompt
        return prepared

    def _weight_name(self, layer):
        return f"{self.hparams.rewrite_module_tmp.format(layer)}.weight"
