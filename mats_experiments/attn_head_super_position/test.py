# %%

%load_ext autoreload
%autoreload 2

# %%
from data import select_token_range, big_data


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

from attention_attribution import (
    get_attn_attrib_on_seq,
    get_bos_ablate_for_head,
    get_attn_attrib,
    AblationType,
    GG,
)

# %%
import torch
import einops

# %%
einops.rearrange(torch.tensor([[1, 2], [3, 4]]), "a b -> (a b)")



# %%
big_data


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
    "device": "cuda:1",
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
test_tokens_for_bos_ablate = select_token_range(0, 100).to(cfg["device"])


# %%

bos_ablate_for_head = get_bos_ablate_for_head(
    model,
    test_tokens_for_bos_ablate,
    num_samples=100,
    k=3,
    bos_value_compare_ratio=0.1,
    bos_ablate_threshold=0.75,
)


# %%
bos, zero, mix = get_attn_attrib_on_seq(
    model, test_tokens_for_bos_ablate[:20], 10, bos_ablate_for_head, time_run=True
)

# %%
torch.cuda.device_count()


# %%
test_prompts = [
    # "14. Missouri 15. Alabama 16. Georgia 17",
    # "I went on a walk in the park today and I came across a very large tree"
    "14. Missouri 15. Alabama 16. Kentucky 17",
    "14. Maine 15. Georgia 16. Vermont 17",
    "14. Colorado 15. California 16. Arizona 17",
]
test_prompt_tokens = model.to_tokens(test_prompts)

test_prompt_short = [
    "14. Missouri 15. Alabama 16. Kentucky",
    "14. Maine 15. Georgia 16. Vermont",
    "14. Colorado 15. California 16. Arizona",
]
test_prompt_short_tokens = model.to_tokens(test_prompt_short)

# %%

for ablate_type in AblationType:
    test_attr = get_attn_attrib(
        model, test_prompt_tokens[:1], -2, ablate_type, bos_ablate_for_head
    )

    test_attr_2 = get_attn_attrib(
        model, test_prompt_short_tokens[:1], -1, ablate_type, bos_ablate_for_head
    )

    print(ablate_type, torch.allclose(test_attr, test_attr_2))

# %%
bos, zero, mix = get_attn_attrib_on_seq(
    model, test_prompt_tokens, 1, bos_ablate_for_head, collapse_batch=False
)

for prompt in range(3):
    for ablate_type in AblationType:
        for i in range(1, 4):
            i = -i

            test_attr = get_attn_attrib(
                model, test_prompt_tokens[prompt].unsqueeze(0), i, ablate_type, bos_ablate_for_head
            )

            if ablate_type == AblationType.BOS:
                comp = bos
            elif ablate_type == AblationType.MIX:
                comp = mix
            elif ablate_type == AblationType.ZERO:
                comp = zero
            else:
                continue

            print("Prompt", prompt, "Type:", ablate_type.value, "Seq:", i, torch.allclose(comp[prompt, i], test_attr, atol=1e-5))

        print()


# %%
bos, zero, mix = get_attn_attrib_on_seq(
    model, test_prompt_tokens, 1, bos_ablate_for_head, collapse_batch=False
)



# %%
torch.allclose(mix[1, -1], test_attr, atol=1e-5)

# %%
imshow(test_attr)
# imshow(bos_ablate_for_head)

# %%
imshow(mix[1, -1])

# %%
imshow(bos[1, -1])


# %%
(zero[1, -1] - mix[1, -1]).abs().max()







# %%
# start = time.time()

# bos, zero, mix = get_attn_attrib_on_seq(
#     model, test_tokens[:10], 10, bos_ablate_for_head
# )


print(f"Elapsed: {time.time() - start:.2f}s")

# %%
bos.shape

# %%

torch.cuda.memory_allocated(cfg["device"])


# %%
# All state names are one name
test_prompts = [
    "14. Missouri 15. Alabama 16. Kentucky 17",
    "14. Maine 15. Georgia 16. Vermont 17",
    "14. Colorado 15. California 16. Arizona 17",
]

test_prompt_tokens = model.to_tokens(test_prompts)

test_prompt_short = [
    "14. Missouri 15. Alabama 16. Kentucky",
    "14. Maine 15. Georgia 16. Vermont",
    "14. Colorado 15. California 16. Arizona",
]

test_prompt_short_tokens = model.to_tokens(test_prompt_short)

# %%
attrib_1 = get_attn_attrib(
    model, test_prompt_tokens[:1], -2, AblationType.MIX, bos_ablate_for_head
)


attrib_2 = get_attn_attrib(
    model, test_prompt_short_tokens[:1], -1, AblationType.MIX, bos_ablate_for_head
)


cache = GG['cache']
last = cache[list(cache)[0]]
last.shape

# %%
last[0, :, 0]


# %%

torch.allclose(attrib_1, attrib_2)

# %%

(attrib_1 - attrib_2).abs().max()

# %%

attrib_2

# %%
cache = GG['cache']
last = cache[list(cache)[0]]
last.shape

# %%
