# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import torch as t
import einops
import circuitsvis as cv
import plotly.express as px
import time
import tqdm

from torch import Tensor
from transformer_lens import HookedTransformer, utils, ActivationCache
from datasets import load_dataset
from typing import Tuple, List
from jaxtyping import Float, Int, Bool
from functools import partial

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

# %%
cfg = {
    "model": "gpt2",
    "device": "cuda:0",
    "enc_dtype": "bf16",
}

model_dtype = DTYPES[cfg["enc_dtype"]]


# %%
model = (
    HookedTransformer.from_pretrained(cfg["model"])
    .to(DTYPES[cfg["enc_dtype"]])
    .to(cfg["device"])
)


# %%
def shuffle_data(all_tokens):
    return all_tokens[torch.randperm(all_tokens.shape[0])]

# %%

num_samples = 1000

big_data = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train", cache_dir="/workspace/cache/")
big_data.set_format(type="torch", columns=["input_ids"])

tokens = big_data.select(range(num_samples))['input_ids']
tokens = einops.rearrange(tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
tokens = shuffle_data(tokens)

# %%
all_tokens = tokens #.to(torch.bfloat16)


# %%

z_filter = (
    lambda name: name.ends_with('z')
)


def get_loss(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    token_logits = (
        logits[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return token_logits


def get_cache_forward_backward(model, tokens, token_index):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(z_filter, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(z_filter, backward_cache_hook, "bwd")

    logits = model(tokens, return_type="logits")

    loss_per_token = -get_loss(logits, tokens)[0, token_index - 1]

    loss_per_token.backward()
    model.reset_hooks()

    return (
        loss_per_token.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


# %%
def get_attn_attrib(cache, grad_cache, bos_ablate_for_head):
    seq_len = cache["z", 0].shape[1]

    bos_stack = torch.stack(
        [
            einops.repeat(
                cache["z", layer][:, 0, ...], "B ... -> B seq ...", seq=seq_len
            )
            for layer in range(model.cfg.n_layers)
        ],
        dim=0,
    )

    z_stack = torch.stack(
        [cache["z", layer] for layer in range(model.cfg.n_layers)], dim=0
    )
    z_grad_stack = torch.stack(
        [grad_cache["z", layer] for layer in range(model.cfg.n_layers)], dim=0
    )

    # z_patch = t.zeros_like(z_stack)
    z_patch = bos_stack

    # for layer, head in torch.nonzero(t.logical_not(bos_ablate_for_head)).tolist():
    #     z_patch[layer, :, :, head] = bos_stack[layer, :, :, head]

    attn_attr = (z_patch - z_stack) * z_grad_stack

    for layer, head in torch.nonzero(t.logical_not(bos_ablate_for_head)).tolist():
        attn_attr[layer, :, :, head] = torch.zeros_like(attn_attr[layer, :, :, head])

    attn_attr = einops.reduce(
        attn_attr, "layer batch seq head d_head -> layer head", "sum"
    )

    return attn_attr