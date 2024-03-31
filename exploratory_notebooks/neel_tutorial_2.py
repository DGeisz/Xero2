# %%
import plotly.io as pio

pio.renderers.default = "png"

# %%
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

# %%
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

t = torch

# %%
t.set_grad_enabled(False)

# %%
import transformer_lens.patching as patching
from neel_plotly import line, imshow, scatter

# %%
model = HookedTransformer.from_pretrained("gpt2-small")

# %%
prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor(
    [
        [model.to_single_token(answers[i][j]) for j in range(2)]
        for i in range(len(answers))
    ],
    device=model.cfg.device,
)
print("Answer token indices", answer_token_indices)


# %%
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

# %%
CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff


def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )


print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")

# %%
DO_SLOW_RUNS = True

# %%
resid_pre_act_patch_results = patching.get_act_patch_resid_pre(
    model, corrupted_tokens, clean_cache, ioi_metric
)

# %%
imshow(
    resid_pre_act_patch_results,
    yaxis="Layer",
    xaxis="Position",
    #    labels={"x": "Layer", "y": "Position"},
    x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
    title="resid_pre Activation Patching",
)


# %%
attn_head_out_all_pos_act_patch_results = patching.get_act_patch_attn_head_out_all_pos(
    model, corrupted_tokens, clean_cache, ioi_metric
)
imshow(
    attn_head_out_all_pos_act_patch_results,
    yaxis="Layer",
    xaxis="Head",
    title="attn_head_out Activation Patching (All Pos)",
)

# %%
ALL_HEAD_LABELS = [
    f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)
]
if DO_SLOW_RUNS:
    attn_head_out_act_patch_results = patching.get_act_patch_attn_head_out_by_pos(
        model, corrupted_tokens, clean_cache, ioi_metric
    )
    attn_head_out_act_patch_results = einops.rearrange(
        attn_head_out_act_patch_results, "layer pos head -> (layer head) pos"
    )

# %%
imshow(
    attn_head_out_act_patch_results,
    yaxis="Head Label",
    xaxis="Pos",
    x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
    y=ALL_HEAD_LABELS,
    title="attn_head_out Activation Patching By Pos",
)


# %%
attn_head_out_all_pos_act_patch_results

# %%

# %%
