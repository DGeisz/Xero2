# %%
%load_ext autoreload
%autoreload 2

# %%
!wget https://github.com/callummcdougall/sae_visualizer/archive/refs/heads/main.zip
!unzip /content/main.zip

# %%

from pathlib import Path
root = Path("sae_visualizer-main").rename("sae_visualizer")

# %%
%pip install transformer_lens
%pip install plotly
%pip install git+https://github.com/callummcdougall/eindex.git


# %%
from IPython.display import display, HTML, clear_output
clear_output()

# %%
from transformer_lens import HookedTransformer, utils
import torch
from datasets import load_dataset
from typing import Dict
from tqdm.notebook import tqdm
import plotly.express as px
import json

# %%
from torch import nn


# %%
from sae_visualizer.model_fns import AutoEncoderConfig, AutoEncoder
from sae_visualizer.data_fns import get_feature_data, FeatureData
# %%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()




# %%
import time
import gzip
import json
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass
from transformer_lens import utils, HookedTransformer
from transformer_lens.hook_points import HookPoint
import torch
from torch import Tensor
from eindex import eindex
from IPython.display import display, HTML
from typing import Optional, List, Dict, Callable, Tuple, Union, Literal
from dataclasses import dataclass
import torch.nn.functional as F
import einops
from jaxtyping import Float, Int
from collections import defaultdict
from functools import partial
from rich import print as rprint
from rich.table import Table
import pickle
import os

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
encoder.W_enc.shape

# %%
model = HookedTransformer.from_pretrained(encoder.cfg["model_name"]).to(DTYPES[encoder.cfg["enc_dtype"]]).to(encoder.cfg["device"])


# %%
from setup import *

# %%
model.to_str_tokens(numbered_states, prepend_bos=True)

# %%
f_i = 18874

for f_i in indices[:20]:
    numbered_states = '10. Missouri 11. Michigan 12. New Jersey 13. Virginia 14. Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
    # numbered_states = '12. New Jersey 13. Virginia 14. Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
    # numbered_states = 'Virginia 14. Washington 15. California 16. Georgia 17. Pennsylvania 18. Florida 19. Texas 20. New York'
    # numbered_states = 'Washington 15. California 16. Danny 17. Pennsylvania 18. Florida 19. Texas 20. New York'
    # numbered_states = 'Melon 15. Bike 16. Danny 17. Animal 18. Florida 19. Texas 20. New York'

    _, cache = model.run_with_cache(numbered_states)

    z = cache['z', 9]
    z = einops.rearrange(z, "batch seq n_heads d_head -> batch seq (n_heads d_head)")

    loss, r_z, acts, l2_loss, l1_loss = encoder(z)

    feature_acts = acts.squeeze(0)[:, f_i]


    print("Index:", f_i, "Max Activation:", feature_acts.sort(descending=True).values[:4])
    if feature_acts.max() > 1:
        display(cv.tokens.colored_tokens(tokens=model.to_str_tokens(numbered_states), values=feature_acts))

# %%
str_tokens = model.to_str_tokens(numbered_states)
a = acts.squeeze(0)[:, 18].tolist()

list(zip(str_tokens, a))


# %%
z.shape


# %%
utils.get_act_name('z', 9)



# %%
z.shape

# %%
list(cache.keys())

# %%
encoder.W_dec.norm(dim=1).shape


# %%
feature_index = 5003
feature = encoder.W_dec[feature_index]

dots = einops.einsum(feature, encoder.W_dec, 'd, n d -> n')

# %%
values, indices = dots.sort(descending=True)

# %%
indices[:10]
# values[:10]

# %%
