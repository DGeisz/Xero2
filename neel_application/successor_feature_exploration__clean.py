# This notebook contains code that was cleaned up for viewing/using externally.
# To see the final state of the code when I finished the project, see
# neel_application/successor_feature_exploration__messy.py

# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
import plotly.express as px
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from jaxtyping import Float
from typing import List, Optional, Tuple, Union, Dict
from IPython.display import display
from transformer_lens import (
    utils,
    HookedTransformer,
    ActivationCache,
)
import circuitsvis as cv
from functools import partial
from IPython.display import HTML

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
def show_stats_for_string(seq: str):
    _, cache = model.run_with_cache(seq)

    z = cache['z', 9]
    z = einops.rearrange(z, "batch seq n_heads d_head -> batch seq (n_heads d_head)")

    acts = encoder(z)[2]

    feature_acts = acts[0, :, 18]

    display(cv.tokens.colored_tokens(tokens=model.to_str_tokens(seq), values=feature_acts, max_value=2))

    print("Max Activation:", feature_acts.max().item())
    print("All Activations: ", feature_acts.tolist())
    print("Largest Activations:", feature_acts.sort(descending=True).values[:5].tolist())


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

def normalize_corruption_result(activation, base_activation):
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
        title=f"Feature Activation Difference From Patched {component_title}",
        labels={"x": "Position", "y": "Layer"},
    )

    if return_patch:
        return patched_diff

def run_activation_corruption(clean_tokens, corrupted_tokens, hook_name, component_title, return_patch=False):
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

            patched_diff[layer, position] = normalize_corruption_result(feature_activation, base_activation)

    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    imshow(
        patched_diff,
        x=prompt_position_labels,
        title=f"Feature Activation Difference From Corrupted {component_title}",
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
            patched_diff[layer, position] = normalize_corruption_result(feature_activation, base_activation)

    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    imshow(
        patched_diff,
        x=prompt_position_labels,
        title=f"Feature Activation Difference From Zero Ablated {component_title}",
        labels={"x": "Position", "y": "Layer"},
    )

    if return_patch:
        return patched_diff

run_activation_patching_on_residual_stream = partial(run_activation_patching, hook_name="resid_pre", component_title="Residual Stream")
run_activation_patching_on_attn_output = partial(run_activation_patching, hook_name="attn_out", component_title="Attention Output")
run_activation_patching_on_mlp_output = partial(run_activation_patching, hook_name="mlp_out", component_title="MLP Output")

run_activation_corruption_on_residual_stream = partial(run_activation_corruption, hook_name="resid_pre", component_title="Residual Stream")
run_activation_corruption_on_attn_output = partial(run_activation_corruption, hook_name="attn_out", component_title="Attention Output")
run_activation_corruption_on_mlp_output = partial(run_activation_corruption, hook_name="mlp_out", component_title="MLP Output")


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
        title=f"Feature Activation Difference From Patched {component_title}",
        labels={"x": "Head", "y": "Layer"},
    )

    if return_patch:
        return patched_head_diff


def run_attention_activation_corruption(clean_tokens, corrupted_tokens, patching_function, hook_name, component_title, return_patch=False):
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

            patched_head_diff[layer, head_index] = normalize_corruption_result(feature_activation, base_value)

    imshow(
        patched_head_diff,
        title=f"Feature Activation Difference From Corrupted {component_title}",
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

            head_diff[layer, head_index] = normalize_corruption_result(feature_activation, base_value)

            # head_diff[layer, head_index] = base_value - feature_activation

    imshow(
        head_diff,
        title=f"Feature Activation Difference From Zero-Ablated {component_title}",
        labels={"x": "Head", "y": "Layer"},
    )

    if return_patch:
        return head_diff

run_activation_patching_on_z_output = partial(run_attention_activation_patching, patching_function=patch_head_vector, hook_name="z", component_title="Z Output")
run_activation_patching_on_values = partial(run_attention_activation_patching, patching_function=patch_head_vector, hook_name="v", component_title="Attention Values")
run_activation_patching_on_attn_pattern = partial(run_attention_activation_patching, patching_function=patch_head_pattern, hook_name="attn", component_title="Attention Pattern")

run_activation_corruption_on_z_output = partial(run_attention_activation_corruption, patching_function=patch_head_vector, hook_name="z", component_title="Z Output")
run_activation_corruption_on_values = partial(run_attention_activation_corruption, patching_function=patch_head_vector, hook_name="v", component_title="Attention Values")
run_activation_corruption_on_attn_pattern = partial(run_attention_activation_corruption, patching_function=patch_head_pattern, hook_name="attn", component_title="Attention Pattern")

run_zero_ablation_on_attn_head = partial(run_attention_zero_ablation, patching_function=zero_ablate_head_pattern, hook_name="attn", component_title="Attention Pattern")

# %%
def corrupt_activation(original_comp: Float[torch.Tensor, 'batch pos head_index d_head'], hook, pos, head_index, replace_cache):
    original_comp[:, pos, head_index, :] = replace_cache[hook.name][:, pos, head_index, :]

    return original_comp

def run_attention_activation_corruption_on_head(clean_tokens, corrupted_tokens, heads: List[Tuple[int, int]], return_patch=False):
    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    base_value = get_linear_feature_activation_from_cache(clean_cache, averaged=True)

    head_diff = torch.zeros(
        len(heads), clean_tokens.shape[1], device=device, dtype=torch.float32
    )

    for i, (layer, head_index) in enumerate(heads):
        for position in range(clean_tokens.shape[1]):
            layer_9_z_store = []

            hook_fn = partial(corrupt_activation, head_index=head_index, pos=position, replace_cache=corrupted_cache)
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

            head_diff[i, position] = normalize_corruption_result(feature_activation, base_value)


    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    head_labels = [f"L{layer}H{head}" for layer, head in heads]

    imshow(
        head_diff,
        x=prompt_position_labels,
        y=head_labels,
        title=f"Feature Activation Difference From Corrupted Attention Head",
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

            hook_fn = partial(corrupt_activation, head_index=head_index, pos=position, replace_cache=clean_cache)
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
        title=f"Feature Activation Difference From Patched Attention Head",
        labels={"x": "Position", "y": "Head"},
    )

    if return_patch:
        return head_diff


# %%
def isolate_circuit_hook(
    original_comp: Float[torch.Tensor, 'batch pos head_index d_model'],
    hook,
    allowed_heads: Dict[int, List[int]],
    passthrough_layers: List[int],
    zero_ablated_heads: Dict[int, List[int]] = {},
):
    layer = int(hook.name.split('.')[1])

    if layer in passthrough_layers:
        return original_comp

    layer_heads = allowed_heads.get(layer, [])
    zero_ablated_heads_for_layer = zero_ablated_heads.get(layer, [])

    num_tokens = original_comp.shape[1]

    bos_activations = einops.repeat(original_comp[:, 0, :, :], 'N n_heads d_heads -> N pos n_heads d_heads', pos=num_tokens)

    for head in range(model.cfg.n_heads):
        if head not in layer_heads:
            original_comp[:, :, head, :] = bos_activations[:, :, head, :]

        if head in zero_ablated_heads_for_layer:
            original_comp[:, :, head, :] = t.zeros_like(original_comp[:, :, head, :])


    return original_comp


def exclude_heads_from_head_indices(head_indices):
    all_heads = list(range(model.cfg.n_heads))

    return [head for head in all_heads if head not in head_indices]


def run_isolated_circuit(
    tokens, 
    allowed_heads: Dict[int, List[int]], 
    passthrough_layers: List[int],
    zero_ablated_heads: Dict[int, List[int]] = {},
    return_values=False
):
    _, cache = model.run_with_cache(tokens)

    base_activation = get_linear_feature_activation_from_cache(cache, averaged=False)

    layer_9_z_store = []

    hook_fn = partial(
        isolate_circuit_hook, 
        allowed_heads=allowed_heads, 
        passthrough_layers=passthrough_layers, 
        zero_ablated_heads=zero_ablated_heads
    )

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



    print("Isolated Activation:")
    print("Avg Value:", isolated_activation.mean().item().__round__(2))
    print("All Values:", [float(f"{n:.2f}") for n in isolated_activation.tolist()])
    print()
    print("Base Activation:")
    print("Avg Value:", base_activation.mean().item().__round__(2))
    print("All Values:", [float(f"{n:.2f}") for n in base_activation.tolist()])

    if return_values:
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

    for prompt in prompts:
        print(prompt)
    # print("Example prompt:", prompts[0])
    # print_prompts(prompts)
    print()

    print("Avg Activation:", f"{activation.item():.2f}")
    print("Activations:", [float(f"{n:.2f}") for n in get_linear_feature_activation_from_cache(cache, averaged=False).tolist()])

# %%
def run_full_analysis(clean_tokens, corrupted_tokens):
    run_activation_corruption_on_residual_stream(clean_tokens, corrupted_tokens)
    run_activation_patching_on_residual_stream(clean_tokens, corrupted_tokens)

    run_activation_patching_on_attn_output(clean_tokens, corrupted_tokens)
    run_activation_corruption_on_attn_output(clean_tokens, corrupted_tokens)

    run_activation_patching_on_mlp_output(clean_tokens, corrupted_tokens)
    run_activation_corruption_on_mlp_output(clean_tokens, corrupted_tokens)

    run_activation_patching_on_z_output(clean_tokens, corrupted_tokens, return_patch=True)
    run_activation_corruption_on_z_output(clean_tokens, corrupted_tokens, return_patch=True)

    run_activation_patching_on_values(clean_tokens, corrupted_tokens, return_patch=True)
    run_activation_corruption_on_values(clean_tokens, corrupted_tokens, return_patch=True)

    run_activation_patching_on_attn_pattern(clean_tokens, corrupted_tokens, return_patch=True)
    run_activation_corruption_on_attn_pattern(clean_tokens, corrupted_tokens, return_patch=True)

# %% [markdown]
# # Probing the behavior of the successor feature on various prompts

# %%
show_stats_for_string("14. Washington 15. California 16. Oregon")

# %%
show_stats_for_string("14. Washington 15. California 16. Daniel")

# %%
show_stats_for_string("14. Robert 15. Jeff 16. Daniel")

# %%
show_stats_for_string("14. Jeff 15. then we have much later 16. Sam")

# %%
show_stats_for_string("14. Oregon 15. then we have much later 16. Sam")

# %%
show_stats_for_string("Jeff 15. then we have much later 16. Sam")

# %%
show_stats_for_string("14. March 15. April 16. May")

# %%
show_stats_for_string("14. Washington 15. California 16. Cynthia")


# %% [markdown]
# # These are the prompts that will be used for the analysis

# %%
state_triples = [
    ("Missouri", "Michigan", "Virginia"),
    ("Washington", "California", "Georgia"),
    ("Florida", "Texas", "Idaho"),
    ("Nevada", "Alabama", "Ohio"),
    ("Oregon", "Arizona", "Colorado"),
    ("Connecticut", "Delaware", "Maryland"),
    ("Pennsylvania", "Wisconsin", "Minnesota"),
    ("Indiana", "Iowa", "Illinois"),
    ("Kansas", "Kentucky", "Louisiana"),
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
people_names = [
    "Daniel", 
    "Rob", 
    "Ashley", 
    "Doug",
    "Ellen",
    "Richard",
    "Emily",
    "Sarah",
    "Dave"
]

name_corrupted_prompts = [successor_prompt_format.format(pair[0][0], pair[0][1], pair[1]) for pair in zip(state_triples, people_names)]
name_corrupted_tokens = model.to_tokens(name_corrupted_prompts, prepend_bos=True)

print_prompt_stats(name_corrupted_prompts)
# %%

corrupted_number_prompt = "14. {} 15. {} 15. {}"

number_corrupted_prompts = [corrupted_number_prompt.format(*triple) for triple in state_triples]
number_corrupted_tokens = model.to_tokens(number_corrupted_prompts, prepend_bos=True)

print_prompt_stats(number_corrupted_prompts)

# %%
prefix_number_corrupted_prompt_template = "alpha. {} help. {} 16. {}"

prefix_number_corrupted_prompts = [prefix_number_corrupted_prompt_template.format(*triple) for triple in state_triples]
prefix_number_corrupted_tokens = model.to_tokens(prefix_number_corrupted_prompts, prepend_bos=True)

print_prompt_stats(prefix_number_corrupted_prompts)

# %%
long_distance_prompt = "14. {} 15. then we have much later 16. {}"

## Unique pairs of names
people_names_prefix = [
    ("Daniel", "Sarah"), 
    ("Doug", "Ashley"), 
    ("Dave", "Ellen"),
    ( "Emily", "Richard"),
    ("Rob", "Jeff"),
    ("Olivia", "Sam"),
    ("Phil", "Mia"),
    ("Amelia", "Harper"),
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
long_distance_first_number_corrupted_prompt = "alpha and {} 15. then we have much later 16. {}"

long_distance_first_number_corrupted_prompts = [long_distance_first_number_corrupted_prompt.format(*pair) for pair in people_names_prefix]
long_distance_first_number_corrupted_tokens = model.to_tokens(long_distance_first_number_corrupted_prompts, prepend_bos=True)

print_prompt_stats(long_distance_first_number_corrupted_prompts)

# %%
long_distance_last_number_corrupted_prompt = "14. {} 15. then we have much later 13. {}"

long_distance_last_number_corrupted_prompts = [long_distance_last_number_corrupted_prompt.format(*pair) for pair in people_names_prefix]
long_distance_last_number_corrupted_tokens = model.to_tokens(long_distance_last_number_corrupted_prompts, prepend_bos=True)

print_prompt_stats(long_distance_last_number_corrupted_prompts)


# %%
run_full_analysis(
    long_distance_tokens, 
    long_distance_last_number_corrupted_tokens
)

# %%
run_attention_activation_corruption_on_head(
    long_distance_tokens, 
    long_distance_last_number_corrupted_tokens,
    [
        (1, 5),
        (4, 4), 
        # (4, 11), 
        # (5, 0), 
        # (7, 11), 
        # (9, 1)
    ]
)

# %%
run_attention_activation_patching_on_head(
    long_distance_tokens, 
    long_distance_last_number_corrupted_tokens,
    [
        (1, 5),
        (4, 4), 
        # (4, 11), 
        # (5, 0), 
        # (7, 11), 
        # (9, 1)
    ]
)


# %%
# %%
relevant_attention_heads = {
    1: [3, 5, 8, 10],
    2: [9, 11],
    3: [6, 7, 11],
    4: [4, 11],
    5: [0],
    7: [11],
    9: [1],
}

# %%
important_heads = []

for i in relevant_attention_heads:

    if i > 3:
        break

    for h in relevant_attention_heads[i]:
        important_heads.append((i, h))



# %% 
run_isolated_circuit(
    long_distance_tokens, 
    #clean_tokens,
    relevant_attention_heads,
    [0],
)


# %%
# %%
HTML(visualize_attention_patterns(
    [
        head_index(4, 4),
        head_index(4, 11),
        head_index(5, 0),
        head_index(7, 11),
    ],
    clean_cache,
    clean_tokens[0],
    # long_distance_cache,
    # long_distance_tokens[0],
    "Relevant Attention Heads"
))
