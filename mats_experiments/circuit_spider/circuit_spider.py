import torch
import einops

from z_sae import ZSAE
from mlp_transcoder import SparseTranscoder
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from torch import Tensor
from typing import *

from dataclasses import dataclass


class LayerKey(TypedDict):
    mlp: int
    attn: int


@dataclass
class SequenceActivations:
    component_vectors: Float[Tensor, "comp d_model"]
    component_values: Float[Tensor, "comp 1"]
    component_features: Int[Tensor, "comp 1"]
    component_key: List[LayerKey]


class PromptWeb:
    def __init__(self, spider, prompt):
        self.spider = spider
        self.prompt = prompt
        self.tokens = self.spider.model.to_tokens(prompt)

        _, self.cache = self.spider.model.run_with_cache(self.tokens)

        self._seq_activations = {}

    @property
    def model(self):
        return self.spider.model

    @property
    def z_saes(self):
        return self.spider.z_saes

    @property
    def mlp_transcoders(self):
        return self.spider.mlp_transcoders

    def get_sequence_activation(self, seq_index: int):
        act = self._seq_activations.get(seq_index, None)

        if act is not None:
            return act

        component_keys: List[LayerKey] = []
        vectors = []
        values = []
        features = []

        for layer in range(self.model.cfg.n_layers):
            # First handle attention
            z_sae = self.z_saes[layer]

            layer_z = einops.rearrange(
                self.cache["z", layer][0, seq_index],
                "n_heads d_head -> (n_heads d_head)",
            )
            z_acts = self.spider.z_saes[layer](layer_z)[2]
            z_winner_count = z_acts.nonzero().numel()

            z_values, z_max_features = z_acts.topk(k=z_winner_count)

            z_contributions = z_sae.W_dec[z_max_features.squeeze(0)] * z_values.squeeze(
                0
            ).unsqueeze(-1)
            z_contributions = einops.rearrange(
                z_contributions, "winners (n_head d_head) -> winners n_head d_head"
            )
            z_residual_vectors = einops.einsum(
                z_contributions,
                self.model.W_O[layer],
                "winners n_head d_head, n_head d_head d_model -> winners d_model",
            )

            vectors.append(z_residual_vectors)
            values.append(z_values)
            features.append(z_max_features)

            # Now handle the transcoder
            mlp_transcoder = self.mlp_transcoders[layer]
            mlp_input = self.cache["normalized", layer, "ln2"]

            mlp_acts = mlp_transcoder(mlp_input)[1]
            mlp_winner_count = mlp_acts.nonzero().numel()

            mlp_values, mlp_max_features = mlp_acts.topk(k=mlp_winner_count)

            mlp_residual_vectors = mlp_transcoder.W_dec[
                mlp_max_features.squeeze(0)
            ] * mlp_values.squeeze(0).unsqueeze(-1)

            vectors.append(mlp_residual_vectors)
            values.append(mlp_values)
            features.append(mlp_max_features)

            component_keys.append({"mlp": mlp_winner_count, "attn": z_winner_count})

        component_vectors = torch.cat(vectors, dim=0)
        component_values = torch.cat(values, dim=0)
        component_features = torch.cat(features, dim=0)

        self._seq_activations[seq_index] = SequenceActivations(
            component_vectors=component_vectors,
            component_values=component_values,
            component_features=component_features,
            component_key=component_keys,
        )

        return self._seq_activations[seq_index]


class CircuitSpider:
    def __init__(self, model_name="gpt2-small"):
        self.model = HookedTransformer.from_pretrained(model_name)

        self.z_saes = [
            ZSAE.load_zsae_for_layer(i) for i in range(self.model.cfg.n_layers)
        ]

        self.mlp_transcoders = [
            SparseTranscoder.load_from_hugging_face(i)
            for i in range(self.model.cfg.n_layers)
        ]

    def get_useful_shit(self, prompt: str):
        _, cache = self.model.run_with_cache(self.model.to_tokens(prompt))

        for layer in range(self.model.cfg.n_layers):
            z_acts = self.z_saes[layer](cache["z", layer])
