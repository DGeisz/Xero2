# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
import plotly.express as px
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from jaxtyping import Float
from typing import List, Optional, Tuple, Union, Dict
from IPython.display import display
from transformer_lens import (
    utils,
    HookedTransformer,
    ActivationCache,
)
import circuitsvis as cv
from functools import partial
from IPython.display import HTML

from plotly_utils import *

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# %%
t.set_grad_enabled(False)



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
model = HookedTransformer.from_pretrained(encoder.cfg["model_name"]).to(DTYPES[encoder.cfg["enc_dtype"]]).to(encoder.cfg["device"])




# %%
def get_feature_acts(cache):
    layer_nine_z = cache['z', 9]

    layer_nine_z = einops.rearrange(layer_nine_z, "batch seq n_heads d_head -> batch seq (n_heads d_head)")[:, -1, :].squeeze(1)

    linear_feature_activation = einops.einsum(layer_nine_z - encoder.b_dec, encoder.W_enc, 'batch d_model, d_model feature-> batch feature') + encoder.b_enc

    return linear_feature_activation.relu()

# %%
a = {}

# %%
str(a.get('a', None))




# %%
acts = get_feature_acts(cache)[0]

# %%
# acts[acts.nonzero()]

# %%

# %%
pattern[0, -1, -1].sum()

# %%




# %%
z = cache['z', 9]



# %%
print(v.shape, pattern.shape, z.shape)

# %%

zz = einops.einsum(v, pattern, "b seq n_head d_head, b n_head p_seq seq -> b p_seq n_head d_head")

(zz - z).abs().max()



# %%
v.shape, pattern.shape

# %%

# %%
_, cache = model.run_with_cache(model.to_tokens('14. Colorado 15. Missouri 16. California'))

acts = get_feature_acts(cache)
acts.nonzero().numel()

values, max_features = acts.topk(
    k=acts.nonzero().numel()
)

# %%
model.W_O.shape


# %%
# layer_nine_z = cache['z', 9]
layer_nine_z.shape

# encoder(layer_nine_z)


# %%
acts.shape


# %%
(encoder.W_dec[max_features.squeeze(0)] * values.squeeze(0).unsqueeze(-1)).shape

# %%
encoder.W_dec.shape

# %%
max_features.shape



# %%
encoder.W[:, max_features].shape, values.shape



# %%
r_pre = cache['resid_pre', 9]
ln1_norm = cache['normalized', 9, 'ln1']

# %%
ln1 = model.blocks[9].ln1


# %%
x = r_pre - r_pre.mean(dim=-1, keepdim=True)

scale = (x.pow(2).mean(-1, keepdim=True) + ln1.eps).sqrt()

outout = x / scale

# %%
torch.allclose(ln1_norm, outout)

# %%
r_pre.shape

# %%
ln2_norm = cache['normalized', 9, 'ln2']

# %%
def gelu_new(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


# %%
ln2_norm.shape, model.W_in.shape

# %%
mlp_mid = einops.einsum(ln2_norm, model.W_in[9], "batch seq d_model, d_model d_ff -> batch seq d_ff") + model.b_in[9]
# mlp_mid = mlp_mid.gelu()

# %%
torch.allclose(cache['pre', 9], mlp_mid),torch.allclose(cache['post', 9], gelu_new(mlp_mid))
# (cache['post', 9] - mlp_mid).abs().max()

# %%
model.cfg.act_fn










# %%
model.blocks[0].ln1


# %%
cache['normalized', 9, 'ln1'].shape, model.W_Q[9].shape

ln1_norm = cache['normalized', 9, 'ln1']
w_q = model.W_Q[9]

q = einops.einsum(ln1_norm, w_q, "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head") + model.b_Q[9]

# %%
aq = cache['q', 9]

# %%
torch.allclose(q, aq)


# %%
ln_out = model.blocks[9].ln1(cache['resid_pre', 9])

# %%
torch.allclose(ln_out, ln1_norm)

# %%
model.cfg.use_attn_in

# %%
model.cfg










# %%
i = 0

feature_i = max_features[i]

v = cache['v', 9]
pattern = cache['pattern', 9]

pre_z = einops.einsum(v, pattern, "b p_seq n_head d_head, b n_head seq p_seq -> seq b p_seq n_head d_head")[-1, 0]

better_b = einops.rearrange(encoder.b_dec, "(n_head d_head) -> n_head d_head", n_head=12)
better_w_enc = einops.rearrange(encoder.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12)

feature_act = einops.einsum(pre_z, better_w_enc, "seq n_head d_head, n_head d_head feature -> n_head seq feature")
b_mod = einops.einsum(better_b, better_w_enc, "n_head d_head, n_head d_head feature -> feature")

linear_f = einops.einsum(feature_act, "n_head seq feature -> feature") - b_mod + encoder.b_enc

mm = 1
mnn = None if mm is None else -mm

imshow(feature_act[:, :, feature_i], zmax=mm, zmin=mnn)
print(linear_f[feature_i])
print(b_mod[feature_i])
print(encoder.b_enc[feature_i])

# %%
head = 5
h = 300

imshow(pattern[0, head], height=h)
pre_z = einops.einsum(v, pattern, "b p_seq n_head d_head, b n_head seq p_seq -> b n_head seq p_seq d_head")[0, head].norm(dim=-1)
imshow(pre_z, height=h)


# %%
layer = 4
head = 4

s_pattern = cache['pattern', layer]
s_v = cache['v', layer]

h = 400

imshow(s_pattern[0, head], height=h, title=f"L{layer}H{head} Pattern\n (Not Corrected for Value Norm)")
s_pre_z = einops.einsum(s_v, s_pattern, "b p_seq n_head d_head, b n_head seq p_seq -> b n_head seq p_seq d_head")[0, head].norm(dim=-1)
imshow(s_pre_z, height=h, title=f"L{layer}H{head} Pattern\n (Corrected for Value Norm)")


# print(feature_act.sum())

# %%
zz = einops.einsum(v, pattern, "b p_seq n_head d_head, b n_head seq p_seq -> b seq n_head d_head")

# layer_nine_z = einops.rearrange(zz, "batch seq n_heads d_head -> batch seq (n_heads d_head)")[:, -1, :].squeeze(1)
layer_nine_z = zz[:, -1].squeeze(1)

better_b = einops.rearrange(encoder.b_dec, "(n_head d_head) -> 1 n_head d_head", n_head=12)
better_w_enc = einops.rearrange(encoder.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12)

print((layer_nine_z - better_b).sum())

linear_feature_activation = einops.einsum(
    layer_nine_z - better_b, 
    better_w_enc, 
    'batch n_head d_head, n_head d_head feature-> batch feature') + encoder.b_enc
# linear_feature_activation = einops.einsum(layer_nine_z - encoder.b_dec, encoder.W_enc, 'batch d_model, d_model feature-> batch feature') + encoder.b_enc

print(linear_feature_activation[:, feature_i])

# return linear_feature_activation.relu()

# %%



# %%
# imshow(pattern[0, -1])
# pattern[0, -1]
line(pre_z[:, -1].norm(dim=-1))

# %%
line(v[0, :, -1].norm(dim=-1))

# %%



# %%
