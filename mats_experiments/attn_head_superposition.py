# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
import torch as t
import einops
import circuitsvis as cv
import plotly.express as px
import time

from torch import Tensor
from transformer_lens import HookedTransformer, utils, ActivationCache
from datasets import load_dataset
from typing import Tuple, List
from jaxtyping import Float, Int, Bool


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
    "enc_dtype": "fp32",
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


try:
    print("Attempting to load tokens from disk")
    all_tokens = torch.load("/workspace/data/c4_code_2b_tokens_reshaped.pt")
    all_tokens = shuffle_data(all_tokens)
    print("Loaded data")
except ValueError as e:
    print("Loading data from internet")
    data = load_dataset(
        "NeelNanda/c4-code-tokenized-2b", split="train", cache_dir="/workspace/cache/"
    )
    data.save_to_disk("/workspace/data/c4_code_tokenized_2b.hf")
    data.set_format(type="torch", columns=["tokens"])
    all_tokens = data["tokens"]
    all_tokens.shape

    all_tokens_reshaped = einops.rearrange(
        all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128
    )
    all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
    all_tokens_reshaped = all_tokens_reshaped[
        torch.randperm(all_tokens_reshaped.shape[0])
    ]
    torch.save(all_tokens_reshaped, "/workspace/data/c4_code_2b_tokens_reshaped.pt")


# %%
def get_attn_head_ablation_types(
    model,
    tokens,
    num_samples=100,
    k=3,
    bos_value_compare_ratio=0.1,
    bos_ablate_threshold=0.75,
) -> Bool[t.Tensor, "layer head"]:
    _, L = tokens.shape

    patterns = t.zeros((model.cfg.n_layers, model.cfg.n_heads, L, L))

    for chunk in torch.split(t.arange(num_samples), 10):
        _, cache = model.run_with_cache(
            all_tokens[chunk].to(cfg["device"]), return_type=None
        )

        patterns += t.stack(
            [
                cache["pattern", layer].cpu().sum(dim=0)
                for layer in range(model.cfg.n_layers)
            ],
            dim=0,
        )

        del cache

    patterns /= num_samples

    top_k_sources = t.topk(patterns, dim=-1, k=k).indices
    bos_values = patterns[:, :, :, 0]
    max_values = patterns.max(dim=-1).values

    bos_within_range = (bos_values / max_values) > bos_value_compare_ratio
    bos_in_top_k = torch.any(top_k_sources == 0, dim=-1)

    bos_ablate_for_dest = torch.logical_and(bos_within_range, bos_in_top_k)

    bos_ablate_for_head = (
        bos_ablate_for_dest.int().float().mean(dim=-1) > bos_ablate_threshold
    )

    return bos_ablate_for_head


# %% [markdown]
# # Attribution Patching


# %%
filter_bad_hook_names = (
    lambda name: "_input" not in name and "_result" not in name and "_in" not in name
)


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens


def get_cache_forward_backward(model, tokens, token_index):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_bad_hook_names, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_bad_hook_names, backward_cache_hook, "bwd")

    logits = model(tokens, return_type="logits")

    loss_per_token = -get_log_probs(logits, tokens)[0, token_index - 1]

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

    z_patch = t.zeros_like(z_stack)

    for layer, head in torch.nonzero(t.logical_not(bos_ablate_for_head)).tolist():
        z_patch[layer, :, :, head] = bos_stack[layer, :, :, head]

    attn_attr = (z_patch - z_stack) * z_grad_stack

    attn_attr = einops.reduce(
        attn_attr, "layer batch seq head d_head -> layer head", "sum"
    )

    return attn_attr


# %%
# %%
toks.shape


# %%
model.tokenizer.decode(all_tokens[12])


# %%
# i = 100

# "".join(model.to_str_tokens(all_tokens[i]))


# %%
"".join(model.to_str_tokens(all_tokens[1]))
# %%

# input_str = "When I decided to go walking in the park today, I didn't expect to see a giraffe."
# input_str = "personalise it with your images, text and designs on the front or back.\nDark grey, heather blue and burgundy are just some of the new shades available. Added to which there's your classic black and white to make your designs really stand out.\nAll of our t-shirts are made from 100% cotton. We also have a range of long sleeve tees made from pure organic cotton that are highly recommendable for sensitive skin.\nThe long sleeve t-shirts combine warmth and comfort with style. The cut is designed to fit comfortably while allowing air circulation.\nPersonalising the new long sleeve"
input_str = "Mary and John went to the park.  Mary gave the basket to "
# output ="14. Missouri 15. Alabama 16. Georgia 17"

start = time.time()

for i in range(1):
    toks = model.to_tokens(input_str, prepend_bos=True)

    l, cache, grad_cache = get_cache_forward_backward(model, toks, toks.shape[1] - 6)

print("Time taken: ", time.time() - start)


attn_attr = get_attn_attrib(cache, grad_cache, bos_ablate_for_head)

imshow(attn_attr[:, :])
print(model.to_str_tokens(toks))

# %%

# %%


# %%
model.to_str_tokens(toks)

# %%
cache["z", 0].shape


# %%
logits = model(all_tokens[:2].to(cfg["device"]), return_type="logits")

# %%
logits.shape

loss


# %%
bos_ablate_for_head = get_attn_head_ablation_types(
    model, all_tokens, num_samples=10, bos_value_compare_ratio=0.1
)
imshow(
    bos_ablate_for_head.int(),
    labels={"x": "Head", "y": "Layer"},
    title="BOS Ablation Heads",
)

# %%
bos_ablate_for_head


# %%
bos_ablate_for_head.shape


# %%

_, cache = model.run_with_cache(all_tokens[:10].to(cfg["device"]))

# %%
layer = 4
head = 7

layer_patterns = cache["pattern", layer].mean(dim=0)

cv.attention.attention_pattern(
    layer_patterns[head, :80, :80], tokens=["" for _ in range(80)]
)

# %%
indices = torch.topk(layer_patterns, dim=-1, k=3).indices
bos_value = layer_patterns[:, :, 0]
max_value = layer_patterns.max(dim=-1).values

# %%
bos_within_range = (bos_value / max_value) > 0.5

# %%
bos_in_top_k = torch.any(indices == 0, dim=-1)

# %%
both_true = torch.logical_and(bos_within_range, bos_in_top_k)

# %%
head = 2

bos_in_top_k[head], bos_within_range[head], both_true[head]

# %%
both_sum = both_true.int().sum(dim=-1)

# %%


# %%
both_sum[11]

# %%
tk.shape


# %%
torch.any(tk[11, :10] == 0, dim=-1)


# %%
cv.attention.attention_pattern(
    layer_patterns[2, :40, :40], tokens=["" for _ in range(40)]
)

# %%
layer_patterns[2, :, 0]

# %%
bb = torch.nonzero(t.logical_not(bos_ablate_for_head))

# %%
bb.tolist()


# %%
c = 20

a = torch.rand(c, c, c, c)

# %%
a[:, bb, :].shape


# %%
