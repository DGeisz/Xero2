# %%
import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
import gradio as gr
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
from functools import partial
import tqdm.notebook as tqdm
import plotly.express as px
import pandas as pd
import einops
import torch as t


# %%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class BitchyNetwork(nn.Module):
    def __init__(self, cfg, num_features: int, num_winners: int):
        super().__init__()

        d_mlp = cfg["d_mlp"]
        dtype = DTYPES[cfg["enc_dtype"]]

        self.num_winners = num_winners

        self.W = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(num_features, d_mlp, dtype=dtype)
            )
        )

        self.to(cfg["device"])

    def forward(self, x):
        raw_output = einops.einsum(x, self.W, "n d, f d -> n f")
        winner_indices = t.argsort(raw_output, descending=True, dim=-1)[
            :, : self.num_winners
        ]

        # We want to set all values that weren't winners to 0
        mask = t.zeros_like(raw_output)
        winner_rows = (
            t.arange(winner_indices.size(0)).unsqueeze(1).expand_as(winner_indices)
        )
        mask[winner_rows, winner_indices] = 1

        acts = mask * raw_output

        reconstructed_x = einops.einsum(acts, self.W, "n f, f d -> n d")

        loss = (reconstructed_x.float() - x.float()).pow(2).sum(dim=-1).mean(0)

        return loss, reconstructed_x, acts
