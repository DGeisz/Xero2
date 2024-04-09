# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
import plotly.express as px
import torch as t
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple, Union, Dict
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv
from functools import partial
from IPython.display import HTML, IFrame

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# %%
t.set_grad_enabled(False)


# %%
def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()

# %%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.dtype = dtype
        self.device = cfg["device"]


        self.version = 0
        self.to(cfg["device"])

    def forward(self, x, per_token=False):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc) # [batch_size, d_hidden]
        x_reconstruct = acts @ self.W_dec + self.b_dec # [batch_size, act_size]
        if per_token:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1) # [batch_size]
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1)) # [batch_size]
            loss = l2_loss + l1_loss # [batch_size]
        else:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0) # []
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1).mean(dim=0)) # []
            loss = l2_loss + l1_loss # []
        return loss, x_reconstruct, acts, l2_loss, l1_loss


    @classmethod
    def load_from_hf(cls, version, hf_repo="ckkissane/tinystories-1M-SAES"):
        """
        Loads the saved autoencoder from HuggingFace.
        """

        cfg = utils.download_file_from_hf(hf_repo, f"{version}_cfg.json")
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf(hf_repo, f"{version}.pt", force_is_torch=True))
        return self

# %%
# Layer 9
auto_encoder_run = "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9"
encoder = AutoEncoder.load_from_hf(auto_encoder_run, hf_repo="ckkissane/attn-saes-gpt2-small-all-layers")

# %%
model = HookedTransformer.from_pretrained(encoder.cfg["model_name"]).to(DTYPES[encoder.cfg["enc_dtype"]]).to(encoder.cfg["device"])

# %%
FEATURE_I = 18
FEATURE_BIAS = encoder.b_enc[FEATURE_I]


# %%
def get_linear_feature_activation(layer_nine_z, averaged=True, feature_i=18):
    # layer_nine_z = cache['z', 9]
    layer_nine_z = einops.rearrange(layer_nine_z, "batch seq n_heads d_head -> batch seq (n_heads d_head)")[:, -1, :].squeeze(1)
    feature = encoder.W_enc[:, feature_i]

    linear_feature_activation = einops.einsum(layer_nine_z - encoder.b_dec, feature, 'batch d_model, d_model -> batch') + encoder.b_enc[feature_i]

    if averaged:
        return linear_feature_activation.mean(dim=0)
    else:
        return linear_feature_activation

def get_linear_feature_activation_from_cache(cache, averaged=True, feature_i=18):
    layer_nine_z = cache['z', 9]

    return get_linear_feature_activation(layer_nine_z, averaged, feature_i)


# %%
def print_prompts(prompts):
    for prompt in prompts:
        str_tokens = model.to_str_tokens(prompt)
        print("Prompt length:", len(str_tokens))
        print("Prompt as tokens:", str_tokens)



# %%

def get_normalizing_function_for_tokens(clean_tokens=None, corrupted_tokens=None, clean_cache=None, corrupted_cache=None):
    if clean_cache is None:
        if clean_tokens is None:
            raise ValueError("clean_tokens or clean_cache must be provided")

        _, clean_cache = model.run_with_cache(clean_tokens)

    if corrupted_cache is None:
        if corrupted_tokens is None:
            raise ValueError("corrupted_tokens or corrupted_cache must be provided")

        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_activation = get_linear_feature_activation_from_cache(clean_cache, averaged=True)
    corrupted_activation = get_linear_feature_activation_from_cache(corrupted_cache, averaged=True)

    def normalize_activation(activation):
        return (activation - corrupted_activation) / (clean_activation - corrupted_activation)
    
    return normalize_activation

def normalize_ablation_result(activation, base_activation):
    return (base_activation - activation) / (base_activation - FEATURE_BIAS)


# %% 
# Activation patching

def patch_activation(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    clean_cache,
):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component


def zero_ablate_hook(
    activation,
    hook,
    pos,
):
    activation[:, pos, :] = t.zeros_like(activation[:, pos, :])

    return activation

def fetch_layer_9_z_activation(
    activation,
    hook,
    store
):
    store.append(activation)

    return activation

def run_activation_patching(clean_tokens, corrupted_tokens, hook_name, component_title, return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)

    normalize = get_normalizing_function_for_tokens(clean_cache=clean_cache, corrupted_tokens=corrupted_tokens)

    patched_diff = torch.zeros(
        model.cfg.n_layers, clean_tokens.shape[1], device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for position in range(clean_tokens.shape[1]):
            layer_9_z_store = []

            hook_fn = partial(patch_activation, pos=position, clean_cache=clean_cache)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (utils.get_act_name(hook_name, layer), hook_fn),
                    (utils.get_act_name("z", 9), fetch_fn),
                ],
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            patched_diff[layer, position] = normalize(feature_activation)

    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    imshow(
        patched_diff,
        x=prompt_position_labels,
        title=f"Logit Difference From Patched {component_title}",
        labels={"x": "Position", "y": "Layer"},
    )

    if return_patch:
        return patched_diff

def run_activation_ablation(clean_tokens, corrupted_tokens, hook_name, component_title, return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    base_activation = get_linear_feature_activation_from_cache(clean_cache, averaged=True)

    patched_diff = torch.zeros(
        model.cfg.n_layers, clean_tokens.shape[1], device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for position in range(clean_tokens.shape[1]):
            layer_9_z_store = []

            # Basically just reverse the function as for activation patching
            hook_fn = partial(patch_activation, pos=position, clean_cache=corrupted_cache)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (utils.get_act_name(hook_name, layer), hook_fn),
                    (utils.get_act_name("z", 9), fetch_fn),
                ],
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            patched_diff[layer, position] = normalize_ablation_result(feature_activation, base_activation)

    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    imshow(
        patched_diff,
        x=prompt_position_labels,
        title=f"Logit Difference From Ablated {component_title}",
        labels={"x": "Position", "y": "Layer"},
    )

    if return_patch:
        return patched_diff

def run_activation_zero_ablation(clean_tokens, hook_name, component_title, return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)

    base_activation = get_linear_feature_activation_from_cache(clean_cache, averaged=True)

    patched_diff = torch.zeros(
        model.cfg.n_layers, clean_tokens.shape[1], device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for position in range(clean_tokens.shape[1]):
            layer_9_z_store = []

            hook_fn = partial(zero_ablate_hook, pos=position)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (utils.get_act_name(hook_name, layer), hook_fn),
                    (utils.get_act_name("z", 9), fetch_fn),
                ],
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            # patched_diff[layer, position] = (base_activation - feature_activation)
            patched_diff[layer, position] = normalize_ablation_result(feature_activation, base_activation)

    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    imshow(
        patched_diff,
        x=prompt_position_labels,
        title=f"Logit Difference From Zero Ablated {component_title}",
        labels={"x": "Position", "y": "Layer"},
    )

    if return_patch:
        return patched_diff

run_activation_patching_on_residual_stream = partial(run_activation_patching, hook_name="resid_pre", component_title="Residual Stream")
run_activation_patching_on_attn_output = partial(run_activation_patching, hook_name="attn_out", component_title="Attention Output")
run_activation_patching_on_mlp_output = partial(run_activation_patching, hook_name="mlp_out", component_title="MLP Output")

run_activation_ablation_on_residual_stream = partial(run_activation_ablation, hook_name="resid_pre", component_title="Residual Stream")
run_activation_ablation_on_attn_output = partial(run_activation_ablation, hook_name="attn_out", component_title="Attention Output")
run_activation_ablation_on_mlp_output = partial(run_activation_ablation, hook_name="mlp_out", component_title="MLP Output")


run_zero_ablation_on_residual_stream = partial(run_activation_zero_ablation, hook_name="resid_pre", component_title="Residual Stream")
run_zero_ablation_on_attn_output = partial(run_activation_zero_ablation, hook_name="attn_out", component_title="Attention Output")
run_zero_ablation_on_mlp_output = partial(run_activation_zero_ablation, hook_name="mlp_out", component_title="MLP Output")


# %%
def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    head_index,
    clean_cache,
):
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][
        :, :, head_index, :
    ]
    return corrupted_head_vector

def zero_ablate_head_vector(
    head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    head_index,
):
    head_vector[:, :, head_index, :] = t.zeros_like(head_vector[:, :, head_index, :])

    return head_vector

def patch_head_pattern(
    corrupted_head_pattern: Float[torch.Tensor, "batch head_index query_pos d_head"],
    hook,
    head_index,
    clean_cache,
):
    corrupted_head_pattern[:, head_index, :, :] = clean_cache[hook.name][
        :, head_index, :, :
    ]
    return corrupted_head_pattern

def zero_ablate_head_pattern(
    head_pattern: Float[torch.Tensor, "batch head_index query_pos d_head"],
    hook,
    head_index,
):
    head_pattern[:, head_index, :, :] = t.zeros_like(head_pattern[:, head_index, :, :])

    return head_pattern


def run_attention_activation_patching(clean_tokens, corrupted_tokens, patching_function, hook_name, component_title, return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)

    normalize = get_normalizing_function_for_tokens(clean_cache=clean_cache, corrupted_tokens=corrupted_tokens)

    patched_head_diff = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
    )

    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
            layer_9_z_store = []

            hook_fn = partial(patching_function, head_index=head_index, clean_cache=clean_cache)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (utils.get_act_name(hook_name, layer, 'attn'), hook_fn),
                    (utils.get_act_name('z', 9), fetch_fn)
                ],
                return_type="logits",
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            patched_head_diff[layer, head_index] = normalize(
                feature_activation
            )

    imshow(
        patched_head_diff,
        title=f"Logit Difference From Patched {component_title}",
        labels={"x": "Head", "y": "Layer"},
    )

    if return_patch:
        return patched_head_diff


def run_attention_activation_ablation(clean_tokens, corrupted_tokens, patching_function, hook_name, component_title, return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    base_value = get_linear_feature_activation_from_cache(clean_cache, averaged=True)


    patched_head_diff = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
    )

    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
            layer_9_z_store = []

            hook_fn = partial(patching_function, head_index=head_index, clean_cache=corrupted_cache)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (utils.get_act_name(hook_name, layer, 'attn'), hook_fn),
                    (utils.get_act_name('z', 9), fetch_fn)
                ],
                return_type="logits",
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            patched_head_diff[layer, head_index] = normalize_ablation_result(feature_activation, base_value)

    imshow(
        patched_head_diff,
        title=f"Logit Difference From Ablated {component_title}",
        labels={"x": "Head", "y": "Layer"},
    )

    if return_patch:
        return patched_head_diff


def run_attention_zero_ablation(clean_tokens, patching_function, hook_name, component_title, return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)

    base_value = get_linear_feature_activation_from_cache(clean_cache, averaged=True)

    head_diff = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
    )

    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
            layer_9_z_store = []

            hook_fn = partial(patching_function, head_index=head_index)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (utils.get_act_name(hook_name, layer, 'attn'), hook_fn),
                    (utils.get_act_name('z', 9), fetch_fn)
                ],
                return_type="logits",
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            head_diff[layer, head_index] = normalize_ablation_result(feature_activation, base_value)

            # head_diff[layer, head_index] = base_value - feature_activation

    imshow(
        head_diff,
        title=f"Logit Difference From Zero-Ablated {component_title}",
        labels={"x": "Head", "y": "Layer"},
    )

    if return_patch:
        return head_diff

run_activation_patching_on_z_output = partial(run_attention_activation_patching, patching_function=patch_head_vector, hook_name="z", component_title="Z Output")
run_activation_patching_on_values = partial(run_attention_activation_patching, patching_function=patch_head_vector, hook_name="v", component_title="Attention Values")
run_activation_patching_on_attn_pattern = partial(run_attention_activation_patching, patching_function=patch_head_pattern, hook_name="attn", component_title="Attention Pattern")

run_activation_ablation_on_z_output = partial(run_attention_activation_ablation, patching_function=patch_head_vector, hook_name="z", component_title="Z Output")
run_activation_ablation_on_values = partial(run_attention_activation_ablation, patching_function=patch_head_vector, hook_name="v", component_title="Attention Values")
run_activation_ablation_on_attn_pattern = partial(run_attention_activation_ablation, patching_function=patch_head_pattern, hook_name="attn", component_title="Attention Pattern")


# run_zero_ablation_on_z_output = partial(run_attention_zero_ablation, patching_function=zero_ablate_head_vector, hook_name="z", component_title="Z Output")
# run_zero_ablation_on_values = partial(run_attention_zero_ablation, patching_function=zero_ablate_head_vector, hook_name="v", component_title="Attention Values")
# run_zero_ablation_on_attn_pattern = partial(run_attention_zero_ablation, patching_function=zero_ablate_head_pattern, hook_name="attn", component_title="Attention Pattern")
run_zero_ablation_on_attn_head = partial(run_attention_zero_ablation, patching_function=zero_ablate_head_pattern, hook_name="attn", component_title="Attention Pattern")

# %%
def ablate_activation(original_comp: Float[torch.Tensor, 'batch pos head_index d_head'], hook, pos, head_index, replace_cache):
    original_comp[:, pos, head_index, :] = replace_cache[hook.name][:, pos, head_index, :]

    return original_comp

def run_attention_activation_ablation_on_head(clean_tokens, corrupted_tokens, heads: List[Tuple[int, int]], return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    base_value = get_linear_feature_activation_from_cache(clean_cache, averaged=True)

    head_diff = torch.zeros(
        len(heads), clean_tokens.shape[1], device=device, dtype=torch.float32
    )

    for i, (layer, head_index) in enumerate(heads):
        for position in range(clean_tokens.shape[1]):
            layer_9_z_store = []

            hook_fn = partial(ablate_activation, head_index=head_index, pos=position, replace_cache=corrupted_cache)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (utils.get_act_name("z", layer, 'attn'), hook_fn),
                    (utils.get_act_name('z', 9), fetch_fn)
                ],
                return_type="logits",
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            head_diff[i, position] = normalize_ablation_result(feature_activation, base_value)


    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    head_labels = [f"L{layer}H{head}" for layer, head in heads]

    imshow(
        head_diff,
        x=prompt_position_labels,
        y=head_labels,
        title=f"Logit Difference From Ablated Attention Head",
        labels={"x": "Position", "y": "Head"},
    )

    if return_patch:
        return head_diff

def run_attention_activation_patching_on_head(clean_tokens, corrupted_tokens, heads: List[Tuple[int, int]], return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)

    normalize = get_normalizing_function_for_tokens(clean_cache=clean_cache, corrupted_tokens=corrupted_tokens)

    head_diff = torch.zeros(
        len(heads), clean_tokens.shape[1], device=device, dtype=torch.float32
    )

    for i, (layer, head_index) in enumerate(heads):
        for position in range(clean_tokens.shape[1]):
            layer_9_z_store = []

            hook_fn = partial(ablate_activation, head_index=head_index, pos=position, replace_cache=clean_cache)
            fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

            model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (utils.get_act_name("z", layer, 'attn'), hook_fn),
                    (utils.get_act_name('z', 9), fetch_fn)
                ],
                return_type="logits",
            )

            feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

            head_diff[i, position] = normalize(feature_activation)


    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    head_labels = [f"L{layer}H{head}" for layer, head in heads]

    imshow(
        head_diff,
        x=prompt_position_labels,
        y=head_labels,
        title=f"Logit Difference From Patched Attention Head",
        labels={"x": "Position", "y": "Head"},
    )

    if return_patch:
        return head_diff

# %%
def isolate_circuit_hook(
    original_comp: Float[torch.Tensor, 'batch pos head_index d_model'],
    hook,
    allowed_heads: Dict[int, List[int]],
    passthrough_layers: List[int]
):
    layer = int(hook.name.split('.')[1])

    if layer in passthrough_layers:
        return original_comp

    layer_heads = allowed_heads.get(layer, [])

    for head in range(model.cfg.n_heads):
        if head not in layer_heads:
            original_comp[:, :, head, :] = t.zeros_like(original_comp[:, :, head, :])

    return original_comp


def exclude_heads_from_head_indices(head_indices):
    all_heads = list(range(model.cfg.n_heads))

    return [head for head in all_heads if head not in head_indices]


def run_isolated_circuit(
    tokens, 
    allowed_heads: Dict[int, List[int]], 
    passthrough_layers: List[int]
):
    _, cache = model.run_with_cache(tokens)

    base_activation = get_linear_feature_activation_from_cache(cache, averaged=False)

    layer_9_z_store = []

    hook_fn = partial(isolate_circuit_hook, allowed_heads=allowed_heads, passthrough_layers=passthrough_layers)
    fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (lambda name: name.startswith('blocks') and name.endswith('hook_z'), hook_fn),
            (utils.get_act_name('z', 9), fetch_fn)
        ],
        return_type="logits",
    )

    isolated_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=False)

    return isolated_activation, base_activation


# %%
def head_index(layer: int, head: int):
    return (layer * model.cfg.n_heads) + head

def visualize_attention_patterns(
    heads: Union[List[int], int, Float[torch.Tensor, "heads"]],
    local_cache: ActivationCache,
    local_tokens: torch.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[torch.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][batch_index, head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = model.to_str_tokens(local_tokens)

    # Combine the patterns into a single tensor
    patterns: Float[torch.Tensor, "head_index dest_pos src_pos"] = torch.stack(
        patterns, dim=0
    )

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = cv.attention.attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"






# %%
def create_head_patching_scatter_plot(diff_x, diff_y, x_name, y_name):
    head_labels = [
        f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]
    scatter(
        x=utils.to_numpy(diff_x.flatten()),
        y=utils.to_numpy(diff_y.flatten()),
        hover_name=head_labels,
        xaxis=f"{x_name} Patch",
        yaxis=f"{y_name} Patch",
        title=f"Scatter Plot of {x_name} Patching vs {x_name} Patching",
        color=einops.repeat(
            np.arange(model.cfg.n_layers), "layer -> (layer head)", head=model.cfg.n_heads
        ),
    )

pattern_output_scatter_plot = partial(create_head_patching_scatter_plot, x_name="Attention", y_name="Output")
values_output_scatter_plot = partial(create_head_patching_scatter_plot, x_name="Value", y_name="Output")

# %%
def print_prompt_stats(prompts, model=model):
    tokens = model.to_tokens(prompts, prepend_bos=True)

    _, cache = model.run_with_cache(tokens)

    activation = get_linear_feature_activation_from_cache(cache, averaged=True)

    print_prompts(clean_prompts)
    print()

    print("Activations:", activation.item(), get_linear_feature_activation_from_cache(cache, averaged=False))

# %%
def run_full_analysis(clean_tokens, corrupted_tokens):
    run_activation_ablation_on_residual_stream(clean_tokens, corrupted_tokens)
    run_activation_patching_on_residual_stream(clean_tokens, corrupted_tokens)

    run_activation_patching_on_attn_output(clean_tokens, corrupted_tokens)
    run_activation_ablation_on_attn_output(clean_tokens, corrupted_tokens)

    run_activation_patching_on_mlp_output(clean_tokens, corrupted_tokens)
    run_activation_ablation_on_mlp_output(clean_tokens, corrupted_tokens)

    run_activation_patching_on_z_output(clean_tokens, corrupted_tokens, return_patch=True)
    run_activation_ablation_on_z_output(clean_tokens, corrupted_tokens, return_patch=True)

    run_activation_patching_on_values(clean_tokens, corrupted_tokens, return_patch=True)
    run_activation_ablation_on_values(clean_tokens, corrupted_tokens, return_patch=True)

    run_activation_patching_on_attn_pattern(clean_tokens, corrupted_tokens, return_patch=True)
    run_activation_ablation_on_attn_pattern(clean_tokens, corrupted_tokens, return_patch=True)






# %%
state_triples = [
    ("Missouri", "Michigan", "Virginia"),
    ("Washington", "California", "Georgia"),
    ("Florida", "Texas", "Idaho"),
    ("Nevada", "Alabama", "Ohio"),

]

successor_prompt_format = "14. {} 15. {} 16. {}"

clean_prompts = [successor_prompt_format.format(*triple) for triple in state_triples]
clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True)

_, clean_cache = model.run_with_cache(clean_tokens)

print_prompt_stats(clean_prompts)
# %%

random_prefix_corrupted_prompts = [successor_prompt_format.format('and', 'the', triple[2]) for triple in state_triples]
random_prefix_corrupted_tokens = model.to_tokens(random_prefix_corrupted_prompts, prepend_bos=True)

print_prompt_stats(random_prefix_corrupted_prompts)

# %%
people_names = ["Daniel", "Rob", "Ashley", "Doug"]

name_corrupted_prompts = [successor_prompt_format.format(pair[0][0], pair[0][1], pair[1]) for pair in zip(state_triples, people_names)]
name_corrupted_tokens = model.to_tokens(name_corrupted_prompts, prepend_bos=True)

print_prompt_stats(name_corrupted_prompts)


# %%

corrupted_number_prompt = "14. {} 15. {} 15. {}"

number_corrupted_prompts = [corrupted_number_prompt.format(*triple) for triple in state_triples]
number_corrupted_tokens = model.to_tokens(number_corrupted_prompts, prepend_bos=True)

print_prompt_stats(number_corrupted_prompts)

# %%
corrupted_period_prompt = "alpha. {} help. {} 16. {}"

period_corrupted_prompts = [corrupted_period_prompt.format(*triple) for triple in state_triples]
period_corrupted_tokens = model.to_tokens(period_corrupted_prompts, prepend_bos=True)

print_prompt_stats(period_corrupted_prompts)

# %%
long_distance_prompt = "14. {} 15. then we have much later 16. {}"

people_names_prefix = [
    ("Daniel", "Sarah"), 
    ("Doug", "Ashley"), 
    ("Dave", "Ellen"),
    ( "Richard", "Emily")
]

long_distance_prompts = [long_distance_prompt.format(*pair) for pair in people_names_prefix]
long_distance_tokens = model.to_tokens(long_distance_prompts, prepend_bos=True)

_, long_distance_cache = model.run_with_cache(long_distance_tokens)

print_prompt_stats(long_distance_prompts)

# %%
long_distance_prefix_name_corrupted_prompt = "14. and 15. then we have much later 16. {}"

long_distance_prefix_name_corrupted_prompts = [long_distance_prefix_name_corrupted_prompt.format(pair[1]) for pair in people_names_prefix]
long_distance_prefix_name_corrupted_tokens = model.to_tokens(long_distance_prefix_name_corrupted_prompts, prepend_bos=True)

print_prompt_stats(long_distance_prefix_name_corrupted_prompts)

# %%
long_distance_second_number_corrupted_prompt = "14. {} beta. then we have much later 16. {}"

long_distance_second_number_corrupted_prompts = [long_distance_second_number_corrupted_prompt.format(*pair) for pair in people_names_prefix]
long_distance_second_number_corrupted_tokens = model.to_tokens(long_distance_second_number_corrupted_prompts, prepend_bos=True)

print_prompt_stats(long_distance_second_number_corrupted_prompts)
# %%
long_distance_first_number_corrupted_prompt = "{} 15. then we have much later 16. {}"

long_distance_first_number_corrupted_prompts = [long_distance_first_number_corrupted_prompt.format(*pair) for pair in people_names_prefix]
long_distance_first_number_corrupted_tokens = model.to_tokens(long_distance_first_number_corrupted_prompts, prepend_bos=True)

print_prompt_stats(long_distance_first_number_corrupted_prompts)

# %%
months = [
    "January", "February", "March", "April" 
]

simple_prompt = "the weekend of {} 16-"

simple_prompts = [simple_prompt.format(month) for month in months]
simple_tokens = model.to_tokens(simple_prompts, prepend_bos=True)

_, simple_cache = model.run_with_cache(simple_tokens)

print_prompt_stats(simple_prompts)

# %%
# simple_corrupted_prompts = ["the weekend of but 16-" for _ in range(len(simple_prompts))]
simple_corrupted_prompts = [f"the weekend of {month} 15-" for month in months]
simple_corrupted_tokens = model.to_tokens(simple_corrupted_prompts, prepend_bos=True)

print_prompt_stats(simple_corrupted_prompts)


# %%
run_full_analysis(simple_tokens, simple_corrupted_tokens)

# %%
run_activation_ablation_on_residual_stream(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)
# %%
run_activation_patching_on_residual_stream(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)

# %%
run_activation_patching_on_attn_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)

# %%
run_activation_patching_on_mlp_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)

# %%

run_activation_ablation_on_mlp_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)

# %%
patched_z_diff = run_activation_patching_on_z_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
patched_z_diff = run_activation_ablation_on_z_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_patching_on_values(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_ablation_on_values(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)


# %%
patched_attn_diff = run_activation_patching_on_attn_pattern(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
patched_attn_diff = run_activation_ablation_on_attn_pattern(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
run_attention_activation_ablation_on_head(
    # long_distance_tokens, 
    # long_distance_prefix_name_corrupted_tokens, 

    clean_tokens, 
    # random_prefix_corrupted_tokens, 

    number_corrupted_tokens,
    # long_distance_second_number_corrupted_tokens,
    [
        # (1, 5), 
        # (2, 0), 
        # (4, 4), 
        # (4, 11), 
        # (3, 2), 
        # (5, 0), 
        # (5, 8), 
        # (7, 11), 

        (6, 3),
        (6, 6),
        (6, 9),
        (6, 10)
        # (9, 1)
    ]
)

# %%
run_attention_activation_patching_on_head(
    clean_tokens, 
    # random_prefix_corrupted_tokens, 
    number_corrupted_tokens,
    # long_distance_second_number_corrupted_tokens,
    [
        (6, 3),
        (6, 6),
        (6, 9),
        (6, 10)
        # (1, 5), 
        # (2, 0), 
        # (4, 4), 
        # (4, 11), 
        # (3, 2), 
        # (5, 0), 
        # (5, 8), 
        # (7, 11), 
        # (9, 1)
    ]
)

# %% 
run_isolated_circuit(clean_tokens, {
    # 5: [0, 8],
    # 7: [11],
    # 8: exclude_heads_from_head_indices([2, 3, 4, 5]),
    # 9: exclude_heads_from_head_indices([0, 4, 5, 6, 7, 8, 9, 10, 11])
    # 6: exclude_heads_from_head_indices([0, 1, 2, 4, 5, 7, 8]),
    6: [3, 6, 9, 10],
    7: [11],
    9: [1]
    # 9: [0, 1, 2]
    }, 
    # list(range(10))
    [0, 1, 2, 3, 4, 5]
)
# %%

# run_isolated_circuit(long_distance_tokens, {
#     # 5: [0, 8],
#     # 7: [11],
#     8: exclude_heads_from_head_indices([2, 3, 4, 5]),
#     9: exclude_heads_from_head_indices([0, 4, 5, 6, 7, 8, 9, 10, 11])
#     # 9: [0, 1, 2]
#     }, 
#     # list(range(10))
#     [0, 1, 2, 3, 4, 5, 6, 7, ]
# )

# %%
HTML(visualize_attention_patterns(
    [
        head_index(6, 3),
        head_index(6, 6),
        head_index(6, 9),
        head_index(6, 10),
    ],
    clean_cache,
    clean_tokens[0],
    # long_distance_cache,
    # long_distance_tokens[0],
    "Some cool heads"
))



# %%
HTML(visualize_attention_patterns(
    [
        head_index(1, 5),
        # head_index(2, 0),
        head_index(4, 4),
        head_index(4, 11),
        # head_index(3, 2),
        head_index(5, 0),
        # head_index(5, 8),
        head_index(7, 11),
        head_index(9, 1)
    ],
    # clean_cache,
    # clean_tokens[0],
    long_distance_cache,
    long_distance_tokens[0],
    "Some cool heads"
))



# %%

run_activation_ablation_on_residual_stream(long_distance_tokens, long_distance_second_number_corrupted_tokens)
# %%
run_activation_patching_on_residual_stream(long_distance_tokens, long_distance_second_number_corrupted_tokens)

# %%
run_activation_patching_on_attn_output(long_distance_tokens, long_distance_second_number_corrupted_tokens)

# %%
run_activation_patching_on_mlp_output(long_distance_tokens, long_distance_second_number_corrupted_tokens)

# %%
patched_z_diff = run_activation_patching_on_z_output(long_distance_tokens, long_distance_second_number_corrupted_tokens, return_patch=True)

# %%
patched_z_diff = run_activation_ablation_on_z_output(long_distance_tokens, long_distance_second_number_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_patching_on_values(long_distance_tokens, long_distance_second_number_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_ablation_on_values(long_distance_tokens, long_distance_second_number_corrupted_tokens, return_patch=True)


# %%
patched_attn_diff = run_activation_patching_on_attn_pattern(long_distance_tokens, long_distance_second_number_corrupted_tokens, return_patch=True)



# %%

run_activation_ablation_on_residual_stream(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)
# %%
run_activation_patching_on_residual_stream(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)

# %%
run_activation_patching_on_attn_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)

# %%
run_activation_patching_on_mlp_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens)

# %%
patched_z_diff = run_activation_patching_on_z_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
patched_z_diff = run_activation_ablation_on_z_output(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_patching_on_values(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)

# %%
patched_attn_diff = run_activation_patching_on_attn_pattern(long_distance_tokens, long_distance_prefix_name_corrupted_tokens, return_patch=True)


# %%
run_zero_ablation_on_residual_stream(long_distance_tokens)

# %%
run_zero_ablation_on_attn_output(long_distance_tokens)

# %%
run_zero_ablation_on_mlp_output(long_distance_tokens)

# %%
run_zero_ablation_on_attn_head(long_distance_tokens)



# %%
run_activation_patching_on_residual_stream(clean_tokens, period_corrupted_tokens)

# %%
run_activation_ablation_on_residual_stream(clean_tokens, period_corrupted_tokens)


# %%
run_activation_patching_on_attn_output(clean_tokens, period_corrupted_tokens)

# %%
run_activation_patching_on_mlp_output(clean_tokens, period_corrupted_tokens)

# %%
patched_z_diff = run_activation_patching_on_z_output(clean_tokens, period_corrupted_tokens, return_patch=True)

# %%
patched_z_diff = run_activation_ablation_on_z_output(clean_tokens, period_corrupted_tokens, return_patch=True)


# %%
patched_value_diff = run_activation_patching_on_values(clean_tokens, period_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_ablation_on_values(clean_tokens, period_corrupted_tokens, return_patch=True)

# %%
patched_attn_diff = run_activation_patching_on_attn_pattern(clean_tokens, period_corrupted_tokens, return_patch=True)

# %%
patched_attn_diff = run_activation_ablation_on_attn_pattern(clean_tokens, period_corrupted_tokens, return_patch=True)

# %%
HTML(visualize_attention_patterns(
    [
        head_index(1, 5),
        head_index(2, 0),
        head_index(4, 4),
        head_index(3, 2),
        head_index(5, 0),
        head_index(5, 8),
        head_index(7, 11),
        head_index(9, 1)
    ],
    # clean_cache,
    # clean_tokens[0],
    long_distance_cache,
    long_distance_tokens[0],
    "Some cool heads"
))


# %%
run_zero_ablation_on_attn_head(clean_tokens)

# %%
run_zero_ablation_on_attn_head(clean_tokens[:2, :])

# %%
clean_tokens[:1, :].shape


# # %%
# run_zero_ablation_on_attn_pattern(clean_tokens)

# %%
run_zero_ablation_on_residual_stream(clean_tokens)

# %%
run_zero_ablation_on_mlp_output(clean_tokens)

# %%
run_zero_ablation_on_attn_output(clean_tokens)


# %%
run_activation_patching_on_residual_stream(clean_tokens, random_prefix_corrupted_tokens)

# %%
run_activation_patching_on_attn_output(clean_tokens, random_prefix_corrupted_tokens)

# %%
run_activation_patching_on_mlp_output(clean_tokens, random_prefix_corrupted_tokens)

# %%
patched_z_diff = run_activation_patching_on_z_output(clean_tokens, random_prefix_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_patching_on_values(clean_tokens, random_prefix_corrupted_tokens, return_patch=True)

# %%
patched_attn_diff = run_activation_patching_on_attn_pattern(clean_tokens, random_prefix_corrupted_tokens, return_patch=True)


# %%
run_activation_patching_on_residual_stream(clean_tokens, name_corrupted_tokens)

# %%
run_activation_patching_on_attn_output(clean_tokens, name_corrupted_tokens)

# %%
patched_z_diff = run_activation_patching_on_z_output(clean_tokens, name_corrupted_tokens, return_patch=True)

# %%
patched_value_diff = run_activation_patching_on_values(clean_tokens, name_corrupted_tokens, return_patch=True)

# %%
patched_attn_diff = run_activation_patching_on_attn_pattern(clean_tokens, name_corrupted_tokens, return_patch=True)


# %%
pattern_output_scatter_plot(patched_attn_diff, patched_z_diff)


# %%
values_output_scatter_plot(patched_value_diff, patched_z_diff)

# %%
top_k = 3



top_positive_logit_attr_heads = torch.topk(
    patched_attn_diff.flatten(), k=top_k
).indices

HTML(visualize_attention_patterns(
    # top_positive_logit_attr_heads,
    # [(9 * 12) + 1],
    [
        head_index(1, 5),
        head_index(3, 2),
        head_index(7, 11),
        head_index(5, 0),
        head_index(4, 4)
    ],
    long_distance_cache,
    long_distance_tokens[0],
    f"Top {top_k} Positive Logit Attribution Heads",
))

# %%


# %%

head_labels = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]
scatter(
    x=utils.to_numpy(patched_attn_diff.flatten()),
    y=utils.to_numpy(patched_z_diff.flatten()),
    hover_name=head_labels,
    xaxis="Attention Patch",
    yaxis="Output Patch",
    title="Scatter plot of output patching vs attention patching",
    color=einops.repeat(
        np.arange(model.cfg.n_layers), "layer -> (layer head)", head=model.cfg.n_heads
    ),
)


# %%
scatter(
    x=utils.to_numpy(patched_value_diff.flatten()),
    y=utils.to_numpy(patched_z_diff.flatten()),
    hover_name=head_labels,
    xaxis="Value Patch",
    yaxis="Output Patch",
    title="Scatter plot of output patching vs value patching",
    color=einops.repeat(
        np.arange(model.cfg.n_layers), "layer -> (layer head)", head=model.cfg.n_heads
    ),
)





# %%
a = run_activation_patching_on_mlp_output(clean_tokens, name_corrupted_tokens)

# %%



# %%
utils.get_act_name("z", 0, 'attn')


# %%

patched_attn_diff = torch.zeros(
    model.cfg.n_layers, clean_tokens.shape[1], device=device, dtype=torch.float32
)
patched_mlp_diff = torch.zeros(
    model.cfg.n_layers, clean_tokens.shape[1], device=device, dtype=torch.float32
)

corrupted_tokens = number_corrupted_tokens
# normalize = normalize_activation_against_number_corrupted

for layer in range(model.cfg.n_layers):
    for position in range(clean_tokens.shape[1]):
        layer_9_z_store = []

        hook_fn = partial(patch_activation, pos=position, clean_cache=clean_cache)
        fetch_fn = partial(fetch_layer_9_z_activation, store=layer_9_z_store)

        layer_9_z_store.clear()
        model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[
                (utils.get_act_name("attn_out", layer), hook_fn),
                (utils.get_act_name("z", 9), fetch_fn),
            ],
            return_type="logits",
        )

        attn_feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)

        layer_9_z_store.clear()
        patched_mlp_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[
                (utils.get_act_name("mlp_out", layer), hook_fn),
                (utils.get_act_name("z", 9), fetch_fn),
            ],
            return_type="logits",
        )
        mlp_feature_activation = get_linear_feature_activation(layer_9_z_store[0], averaged=True)


        patched_attn_diff[layer, position] = normalize(
            attn_feature_activation
        )
        patched_mlp_diff[layer, position] = normalize(
            mlp_feature_activation
        )

# %%
imshow(
    patched_attn_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched Attention Layer",
    labels={"x": "Position", "y": "Layer"},
)

# %%
imshow(
    patched_mlp_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched MLP Layer",
    labels={"x": "Position", "y": "Layer"},
)



# %%
position = 1
layer = 1

hook_fn = partial(patch_activation, pos=position, clean_cache=clean_cache)
cache = model.run_with_hooks(
    name_corrupted_tokens,
    fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
    # return_type="cache",
)

# %%



































# %%

# %%
print_prompts(clean_prompts)


# %%
people_names = ["Daniel", "Rob", "Emily", "Sarah", "Chris"]
transport_names = ["Bike", "Plane", "Car", "Boat", "Train"]

fully_corrupted_prompts = [successor_prompt_format.format(*pair) for pair in zip(transport_names, people_names)]
fully_corrupted_prompt_tokens = model.to_tokens(fully_corrupted_prompts, prepend_bos=True)

_, fully_corrupted_cache = model.run_with_cache(fully_corrupted_prompt_tokens)

print("Fully Corrupted Prompts:")
print_prompts(fully_corrupted_prompts)
print()

beginning_corrupted_prompts = [successor_prompt_format.format(pair[0], pair[1][1]) for pair in zip(transport_names, state_triples)]
beginning_corrupted_prompt_tokens = model.to_tokens(beginning_corrupted_prompts, prepend_bos=True)

_, beginning_corrupted_cache = model.run_with_cache(beginning_corrupted_prompt_tokens)

print("Beginning Corrupted Prompts:")
print_prompts(beginning_corrupted_prompts)
print()

ending_corrupted_prompts = [successor_prompt_format.format(pair[0][0], pair[1]) for pair in zip(state_triples, people_names)]
ending_corrupted_prompt_tokens = model.to_tokens(ending_corrupted_prompts, prepend_bos=True)

_, ending_corrupted_cache = model.run_with_cache(ending_corrupted_prompt_tokens)


print("Ending Corrupted Prompts:")
print_prompts(ending_corrupted_prompts)
print()


# %%
get_linear_feature_activation_from_cache(clean_cache, averaged=False)

# %%
get_linear_feature_activation_from_cache(fully_corrupted_cache, averaged=False), get_linear_feature_activation(fully_corrupted_cache)

# %%
get_linear_feature_activation_from_cache(beginning_corrupted_cache, averaged=False), get_linear_feature_activation(beginning_corrupted_cache)















# %%
numbered_states = clean_prompts[1]

# %%
numbered_states



# %%
# numbered_states = '10. Missouri 11. Michigan 12. New Jersey 13. Virginia 14. Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = '12. New Jersey 13. Virginia 14. Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = '14. Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = 'Virginia 14. Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = '14. and 15. then 16. Cynthia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = "Nah Sarah 15. then we have much later 16. Hannah"
# numbered_states = "Sarah salad- 15, after all of that- 16, Charlie better-"
numbered_states = "The weekend of California 16-"
# numbered_states = "14. and 15. then 16. Cynthia"
# numbered_states = 'Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = '15. Girl 16. Boy 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = '15. California 16. Danny 17. Pennsylvania 18. Florida 19. Texas 20. New York'
# numbered_states = 'Melon 15. Bike 16. Danny 17. Animal 18. Florida 19. Texas 20. New York'
# numbered_states = clean_prompts[0]

_, cache = model.run_with_cache(numbered_states)


z = cache['z', 9]
z = einops.rearrange(z, "batch seq n_heads d_head -> batch seq (n_heads d_head)")

attn_out = cache['attn_out', 9]

loss, r_z, acts, l2_loss, l1_loss = encoder(z)

feature_acts = acts[0, :, 18]

print(feature_acts.sort(descending=True).values[:5])

display(cv.tokens.colored_tokens(tokens=model.to_str_tokens(numbered_states), values=feature_acts, max_value=2))

# %%
feature_acts

# %%
res = 0

def get_layer_9_z(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
):
    global res
    res = corrupted_residual_component
    return corrupted_residual_component

clean_tokens = model.to_tokens(numbered_states, prepend_bos=True)


model.run_with_hooks(
    clean_tokens,
    fwd_hooks=[(utils.get_act_name("z", 9), get_layer_9_z)],
)

# %%

loss, r_z, acts, l2_loss, l1_loss = encoder(res)

feature_acts = acts[0, :, 18]

display(cv.tokens.colored_tokens(tokens=model.to_str_tokens(numbered_states), values=feature_acts, max_value=2))

# %%
feature_acts



# %%


patched_residual_stream_diff = torch.zeros(
    model.cfg.n_layers, clean_tokens.shape[1], device=device, dtype=torch.float32
)
for layer in range(model.cfg.n_layers):
    for position in range(clean_tokens.shape[1]):
        hook_fn = partial(patch_activation, pos=position, clean_cache=clean_cache)
        patched_logits = model.run_with_hooks(
            name_corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
            return_type="logits",
        )
        patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

        patched_residual_stream_diff[layer, position] = normalize_patched_logit_diff(
            patched_logit_diff
        )





# %%

# %%
f18 = encoder.W_enc[:, 18]

# %%
feature_acts


# %%
a = einops.einsum(z - encoder.b_dec, f18, 'batch seq d_model, d_model -> batch seq') + encoder.b_enc[18]


# %%
a

# %%
feature_acts




# %%
a

# %%

# %%
