import torch
import einops

from z_sae import ZSAE
from mlp_transcoder import SparseTranscoder
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from torch import Tensor
from typing import *
from tqdm import trange

from dataclasses import dataclass


class LayerKey(TypedDict):
    mlp: int
    attn: int


@dataclass
class ActiveFeatures:
    vectors: Float[Tensor, "comp d_model"]
    values: Float[Tensor, "comp 1"]
    features: Int[Tensor, "comp 1"]
    keys: List[LayerKey]

    def get_vectors_before_comp(self, kind: str, layer: int):
        max_index = 0

        for i in range(layer):
            max_index += self.keys[i]["mlp"] + self.keys[i]["attn"]

        if kind == "mlp":
            max_index += self.keys[layer]["attn"]

        return self.vectors[:max_index]

    @property
    def max_active_features(self):
        lens = []

        for key in self.keys:
            lens.append(key["mlp"])
            lens.append(key["attn"])

        return max(lens)

    def get_top_k_features(self, activations: Float[Tensor, "comp"], k=10):
        values, indices = activations.topk(k=k)

        features = []

        for v, i in zip(values.tolist(), indices.tolist()):
            start_i = 0

            for l, key in enumerate(self.keys):
                if i < start_i + key["attn"]:
                    features.append(("attn", l, self.features[i], v))
                    break

                start_i += key["attn"]

                if i < start_i + key["mlp"]:
                    features.append(("mlp", l, self.features[i], v))
                    break

                start_i += key["mlp"]

        return features

    def get_top_k_labels(self, activations: Float[Tensor, "comp"], k=10):
        features = self.get_top_k_features(activations, k=k)

        return [
            f"{kind.capitalize()} {layer} | Feature: {feature} | Value: {value:.3g}"
            for kind, layer, feature, value in features
        ]

    def reshape_activations_for_visualization(
        self, activations: Float[Tensor, "comp 1"]
    ):
        # assert activations.size(0) == self.vectors.size(0)

        min_val = activations.min()

        visualization = torch.ones(
            (12 * 2, self.max_active_features), device=activations.device
        ) * (min_val / 2)
        start_i = 0

        a_len = activations.size(0)

        for i, key in enumerate(self.keys):
            ii = 2 * i

            if start_i + key["attn"] > a_len:
                break

            visualization[ii, : key["attn"]] = activations[
                start_i : start_i + key["attn"]
            ]
            start_i += key["attn"]

            if start_i + key["mlp"] > a_len:
                break

            visualization[ii + 1, : key["mlp"]] = activations[
                start_i : start_i + key["mlp"]
            ]
            start_i += key["mlp"]

        return visualization


class PromptSpiderWeb:
    def __init__(self, spider, prompt):
        self.spider = spider
        self.prompt = prompt
        self.tokens = self.spider.model.to_tokens(prompt)

        _, self.cache = self.spider.model.run_with_cache(self.tokens)

        self._seq_activations = {}

    @property
    def n_tokens(self):
        return self.tokens.size(1)

    @property
    def model(self):
        return self.spider.model

    @property
    def z_saes(self):
        return self.spider.z_saes

    @property
    def mlp_transcoders(self):
        return self.spider.mlp_transcoders

    def get_active_features(self, seq_index: int):
        if seq_index < 0:
            seq_index += self.n_tokens

        act = self._seq_activations.get(seq_index, None)

        if act is not None:
            return act

        component_keys: List[LayerKey] = []
        vectors = []
        values = []
        features = []

        for layer in trange(self.model.cfg.n_layers):
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
                z_contributions,
                "winners (n_head d_head) -> winners n_head d_head",
                n_head=self.model.cfg.n_heads,
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
            mlp_input = self.cache["normalized", layer, "ln2"][:, seq_index]

            mlp_acts = mlp_transcoder(mlp_input)[1]
            mlp_winner_count = mlp_acts.nonzero().numel()

            mlp_values, mlp_max_features = mlp_acts.topk(k=mlp_winner_count)

            mlp_residual_vectors = mlp_transcoder.W_dec[
                mlp_max_features.squeeze(0)
            ] * mlp_values.squeeze(0).unsqueeze(-1)

            vectors.append(mlp_residual_vectors)
            values.append(mlp_values.squeeze())
            features.append(mlp_max_features.squeeze())

            component_keys.append({"mlp": mlp_winner_count, "attn": z_winner_count})

        component_vectors = torch.cat(vectors, dim=0)
        component_values = torch.cat(values, dim=0)
        component_features = torch.cat(features, dim=0)

        self._seq_activations[seq_index] = ActiveFeatures(
            vectors=component_vectors,
            values=component_values,
            features=component_features,
            keys=component_keys,
        )

        return self._seq_activations[seq_index]

    def visualize_activations(self, active_features, activations, k=None):
        if k is None:
            k = 10

        return active_features.reshape_activations_for_visualization(
            activations
        ), active_features.get_top_k_labels(activations, k=k)

    def get_unembed_lens(self, token_i: int, seq_index: int, visualize=False, k=None):
        active_features = self.get_active_features(seq_index)

        activations = einops.einsum(
            active_features.vectors,
            self.model.W_U[:, token_i],
            "comp d_model, d_model -> comp",
        )

        if visualize:
            return self.visualize_activations(active_features, activations, k=k)
        else:
            return activations

    def get_head_seq_activations_for_z_feature(self, layer: int, feature: int):
        v = self.cache["v", layer]
        pattern = self.cache["pattern", 9]
        encoder = self.z_saes[layer]

        pre_z = einops.einsum(
            v,
            pattern,
            "b p_seq n_head d_head, b n_head seq p_seq -> seq b p_seq n_head d_head",
        )[-1, 0]

        better_w_enc = einops.rearrange(
            encoder.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12
        )[:, :, feature]

        feature_act = einops.einsum(
            pre_z, better_w_enc, "seq n_head d_head, n_head d_head -> n_head seq"
        )

        return feature_act

    def get_sae_feature_lens_on_head_seq(
        self,
        feature: int,
        layer: int,
        head: int,
        seq_index: int,
        visualize=False,
        k=None,
    ):
        active_features = self.get_active_features(seq_index)
        z_sae = self.z_saes[layer]

        vectors = active_features.get_vectors_before_comp("attn", layer)

        effective_v = einops.einsum(
            vectors,
            self.model.W_V[layer, head],
            "comp d_model, d_model d_head -> comp d_head",
        )

        effective_feature = einops.rearrange(
            z_sae.W_enc[:, feature],
            "(n_head d_head) -> n_head d_head",
            n_head=self.model.cfg.n_heads,
        )[head]

        activation = einops.einsum(
            effective_v, effective_feature, "comp d_head, d_head -> comp"
        )

        if visualize:
            return self.visualize_activations(active_features, activation, k=k)
        else:
            return activation


class CircuitSpider:
    def __init__(self, model_name="gpt2-small"):
        self.model = HookedTransformer.from_pretrained(model_name)

        self.z_saes = [
            ZSAE.load_zsae_for_layer(i) for i in trange(self.model.cfg.n_layers)
        ]

        self.mlp_transcoders = [
            SparseTranscoder.load_from_hugging_face(i)
            for i in trange(self.model.cfg.n_layers)
        ]

    def create_prompt_web(self, prompt: str) -> PromptSpiderWeb:
        return PromptSpiderWeb(self, prompt)
