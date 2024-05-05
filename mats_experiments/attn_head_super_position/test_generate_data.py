# %%
%load_ext autoreload
%autoreload 2

# %%

from data import select_token_range, big_data
from generate_data import generate_data

import torch
import torch as t
import einops
import circuitsvis as cv
import plotly.express as px
import time
import tqdm
import boto3

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

from d_types import DTYPES

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
test_tokens_for_bos_ablate = select_token_range(0, 100).to(cfg["device"])

bos_ablate_for_head = get_bos_ablate_for_head(
    model,
    test_tokens_for_bos_ablate,
    num_samples=100,
    k=3,
    bos_value_compare_ratio=0.1,
    bos_ablate_threshold=0.75,
)

# %%
session = boto3.Session(
    # aws_access_key_id='',
    # aws_secret_access_key='',
    region_name="us-east-2",
)
s3 = session.client("s3")

# %%
generate_data(
    s3,
    bos_ablate_for_head,
    num_threads=1,
    seq_batch_size=3,
    num_batches=1
)


# %%
