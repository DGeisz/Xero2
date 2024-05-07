import torch
import torch as t
import einops

from torch import Tensor
from transformer_lens import ActivationCache
from jaxtyping import Float, Int, Bool
from time import time

from enum import Enum


class AblationType(Enum):
    BOS = "bos"
    ZERO = "zero"
    MIX = "mix"
    BOS_MASK_ZERO = "bos_mask_zero"

    def __eq__(self, other):
        if isinstance(other, AblationType):
            return self.value == other.value
        return False


def get_bos_ablate_for_head(
    model,
    tokens,
    num_samples=100,
    k=3,
    bos_value_compare_ratio=0.1,
    bos_ablate_threshold=0.75,
    device="cuda:0",
) -> Bool[torch.Tensor, "layer head"]:
    _, L = tokens.shape

    patterns = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, L, L))

    for chunk in torch.split(torch.arange(num_samples), 10):
        _, cache = model.run_with_cache(tokens[chunk].to(device), return_type=None)

        patterns += torch.stack(
            [
                cache["pattern", layer].cpu().sum(dim=0)
                for layer in range(model.cfg.n_layers)
            ],
            dim=0,
        )

        del cache

    patterns /= num_samples

    top_k_sources = torch.topk(patterns, dim=-1, k=k).indices
    bos_values = patterns[:, :, :, 0]
    max_values = patterns.max(dim=-1).values

    bos_within_range = (bos_values / max_values) > bos_value_compare_ratio
    bos_in_top_k = torch.any(top_k_sources == 0, dim=-1)

    bos_ablate_for_dest = torch.logical_and(bos_within_range, bos_in_top_k)

    bos_ablate_for_head = (
        bos_ablate_for_dest.int().float().mean(dim=-1) > bos_ablate_threshold
    )

    return bos_ablate_for_head


z_filter = lambda name: name.endswith("z")


def get_loss(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logits = logits.log_softmax(dim=-1)
    # Get token loss the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    token_loss = (
        logits[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return token_loss


GG = {}


########################################
# Single Example Attention Attribution #
########################################
def get_cache_forward_backward(model, tokens, token_index):
    _, seq_len = tokens.shape

    if token_index < 0:
        token_index += seq_len

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

    GG["cache"] = grad_cache

    return (
        loss_per_token.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


def get_attn_attrib(
    model, tokens, token_index, ablation_type: AblationType, bos_ablate_for_head
):
    _, cache, grad_cache = get_cache_forward_backward(model, tokens, token_index)

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

    if str(ablation_type) == str(AblationType.ZERO):
        z_patch = torch.zeros_like(z_stack)
    else:
        z_patch = bos_stack

    if str(ablation_type) == str(AblationType.MIX):
        for layer, head in torch.nonzero(t.logical_not(bos_ablate_for_head)).tolist():
            z_patch[layer, :, :, head] = torch.zeros_like(z_patch[layer, :, :, head])

    attn_attr = (z_patch - z_stack) * z_grad_stack

    if str(ablation_type) == str(AblationType.BOS_MASK_ZERO):
        for layer, head in torch.nonzero(
            torch.logical_not(bos_ablate_for_head)
        ).tolist():
            attn_attr[layer, :, :, head] = torch.zeros_like(
                attn_attr[layer, :, :, head]
            )

    attn_attr = einops.reduce(
        attn_attr, "layer batch seq head d_head -> layer head", "sum"
    )

    return attn_attr


########################################
# Multi Example Attention Attribution  #
########################################
def get_z_stack_forward_backward_on_seq(model, tokens, start_index):
    model.reset_hooks()

    _, seq = tokens.shape

    cache_dict = {}

    def forward_cache_hook(act, hook):
        cache_dict[hook.name] = act.detach()

    model.add_hook(z_filter, forward_cache_hook, "fwd")

    grad_cache_helper = {"cache": {}}

    def backward_cache_hook(act, hook):
        grad_cache_helper["cache"][hook.name] = act.detach()

    model.add_hook(z_filter, backward_cache_hook, "bwd")

    logits = model(tokens, return_type="logits")

    cache = ActivationCache(cache_dict, model)

    z_stack = torch.stack(
        [cache["z", layer] for layer in range(model.cfg.n_layers)], dim=0
    )

    bos_stack = torch.stack(
        [
            einops.repeat(cache["z", layer][:, 0, ...], "B ... -> B seq ...", seq=seq)
            for layer in range(model.cfg.n_layers)
        ],
        dim=0,
    )

    z_grad_stack_list = []

    for token_index in range(start_index, seq):
        loss_per_token = -get_loss(logits, tokens)[:, token_index - 1].sum(dim=0)

        retain_graph = token_index != seq - 1
        loss_per_token.backward(retain_graph=retain_graph)

        z_grad_stack = ActivationCache(grad_cache_helper["cache"], model)

        z_grad_stack_list.append(
            torch.stack(
                [z_grad_stack["z", layer] for layer in range(model.cfg.n_layers)]
            )
        )

        grad_cache_helper["cache"] = {}

    model.reset_hooks()

    z_grad_stack = torch.stack(z_grad_stack_list, dim=0)

    return z_stack, z_grad_stack, bos_stack


def get_attn_attrib_on_seq(
    model, tokens, start_index, bos_ablate_for_head, collapse_batch=True, time_run=False
):
    if time_run:
        big_start = time()
        start = big_start
    z_stack, z_grad_stack, bos_stack = get_z_stack_forward_backward_on_seq(
        model, tokens, start_index
    )

    with torch.no_grad():
        if time_run:
            print(f"E1: {time() - start:.2f}s")

        bos_patch = bos_stack
        zero_patch = torch.zeros_like(z_stack)

        mix_patch = bos_stack.clone()

        if time_run:
            start = time()

        for layer, head in torch.nonzero(t.logical_not(bos_ablate_for_head)).tolist():
            mix_patch[layer, :, :, head] = 0

        if time_run:
            print(f"E2: {time() - start:.2f}s")

            start = time()

        bos_attn_attribution = (bos_patch - z_stack).unsqueeze(0) * z_grad_stack
        zero_attn_attribution = (zero_patch - z_stack).unsqueeze(0) * z_grad_stack
        mix_attn_attribution = (mix_patch - z_stack).unsqueeze(0) * z_grad_stack

        if collapse_batch:
            reduction_string = (
                "seq_batch layer batch seq head d_head -> (batch seq_batch) layer head"
            )
        else:
            reduction_string = (
                "seq_batch layer batch seq head d_head -> batch seq_batch layer head"
            )

        bos_attn_attribution = einops.reduce(
            bos_attn_attribution,
            reduction_string,
            "sum",
        )

        zero_attn_attribution = einops.reduce(
            zero_attn_attribution,
            reduction_string,
            "sum",
        )

        mix_attn_attribution = einops.reduce(
            mix_attn_attribution,
            reduction_string,
            "sum",
        )

        if time_run:
            print(f"E3: {time() - start:.2f}s")
            print(f"Total: {time() - big_start:.2f}s")

        return bos_attn_attribution, zero_attn_attribution, mix_attn_attribution
