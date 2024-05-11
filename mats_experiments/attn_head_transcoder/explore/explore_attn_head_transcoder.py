# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import einops

from transformer_lens import HookedTransformer
from plotly_utils import *

# %%


# %%
model = HookedTransformer.from_pretrained('gpt2-small', fold_ln=True)
model.set_use_attn_in(True)

model.set_use_split_qkv_input(True)


# %%
tokens = model.to_tokens("Hi my name is Danny and I like apple pie")

# %%
tokens.shape

# %%
_, cache = model.run_with_cache(tokens)

# %%
model

# %%
list(cache.keys())

# %%
torch.isclose(
cache['blocks.10.ln1.hook_normalized'],
cache['resid_pre', 10],
)

# %%
pre = cache['blocks.10.ln1.hook_normalized']
q = cache['q', 10]

# %%
cache['q_input', 10].shape



# %%
wq = model.W_Q[10]

# %%
pre.shape, wq.shape, q.shape

# %%
cache['attn_in', 10].shape


# %%
imshow((einops.einsum(
    model.blocks[10].ln1(cache['attn_in', 10][:, :, 0, :])
    , wq, "b seq d_model, n_head d_model d_head -> b seq n_head d_head"
) - q)[0, -1])

# %%
imshow(
    q[0, -1]
)

# %%
cache['resid_pre', 8][0, -3].mean()


# %%
(
    einops.einsum(
        model.blocks[10].ln1(cache['attn_in', 10][:, :, 0, :])
        , wq, "b seq d_model, n_head d_model d_head -> b seq n_head d_head") 
    q
).abs().max()

# %%
model.blocks[10].ln1


# %%

(einops.einsum(cache['q_input', 10][:, :, 0, :], wq, "b seq d_model, n_head d_model d_head -> b seq n_head d_head") - q).abs().max()

# %%
cache['resid_pre', 10][0, -1]

# %%
q.shape


# %%
imshow(
    cache['resid_pre', 10][0, -2].reshape(32, -1)
)

# %%
imshow(
    pre[0, -2].reshape(32, -1)
)

# %%
list(cache.keys())

# %%
cache['blocks.10.mlp.hook_pre'].shape, cache['blocks.10.mlp.hook_post'].shape

# %%
model.W_in[0].shape

# %%
(einops.einsum(cache['blocks.10.ln1.hook_normalized'], model.W_in[10], "b seq d_model, d_model d_mlp -> b seq d_mlp") - cache['blocks.10.mlp.hook_pre']).abs().max()

# %%
model.fold_layer_norm

# %%
dir(model)


# %%
