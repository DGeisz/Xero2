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
    batch_size = None
    width = None

    def __init__(self, cfg, num_features: int, num_winners: int, buffer):
        super().__init__()
        self.num_features = num_features

        d_mlp = cfg["d_mlp"]
        self.dtype = DTYPES[cfg["enc_dtype"]]

        self.num_winners = num_winners
        self._buffer = buffer

        self.W = nn.Parameter(
            self._init_weights()
            # torch.nn.init.kaiming_uniform_(
            #     torch.empty(num_features, d_mlp, dtype=dtype)
            # )
        )

        self.to(cfg["device"])

    def random_data_batch(self):
        data = self._buffer.next()

        if self.batch_size is None:
            self.batch_size = data.shape[0]

        return data

    def _init_weights(self):
        self.batch = 0

        if not self.width:
            _, width = self.random_data_batch().shape
            self.width = width

        all_data = []

        for _ in range((self.num_features // self.batch_size) + 1):
            all_data.append(self.random_data_batch())

        features = t.cat(all_data, dim=0)[: self.num_features, : self.width]
        features = features / features.norm(dim=-1, keepdim=True)

        return features.to(self.dtype)

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
