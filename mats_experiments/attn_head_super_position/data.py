import torch
import einops

from datasets import load_dataset


def shuffle_data(all_tokens):
    return all_tokens[torch.randperm(all_tokens.shape[0])]


big_data = load_dataset(
    "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
    split="train",
    cache_dir="/workspace/cache/",
)

big_data.set_format(type="torch", columns=["input_ids"])


def select_token_range(start, num_samples, prepend_bos=True, shuffle=False):
    tokens = big_data.select(range(start, start + num_samples))["input_ids"]
    tokens = einops.rearrange(
        tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128
    )

    if prepend_bos:
        bos = torch.tensor([[50256]]).repeat(tokens.shape[0], 1)
        tokens = torch.cat([bos, tokens], dim=-1)

    if shuffle:
        return shuffle_data(tokens)
    else:
        return tokens
