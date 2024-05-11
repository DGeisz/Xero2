# %%
%load_ext autoreload
%autoreload 2

# %%
from circuit_spider import CircuitSpider
from memory import get_gpu_memory
from plotly_utils import *

# %%
spider = CircuitSpider()


# %%
spider.z_saes[0].W_dec.device

# %%
get_gpu_memory()

# %%
web = spider.create_prompt_web("14. Colorado 15. Michigan 16. Missouri 17")


# %%
web.n_tokens

# %%
web.tokens, web.model.to_str_tokens(web.tokens)


# %%
acts, labels = web.get_sae_feature_lens_on_head_seq(
    1596, 
    9,
    1,
    7,
    visualize=True,
    k=10
)

# %%
labels

# %%
f = web.get_active_features( -2)
# %%
f.keys

# %%
web.cache['ln_final.hook_scale'][:, -3, :]



# %%


# %%
acts.vectors.shape

# %%
spider.model.W_U.shape

# %%
labels = []

for i in range(12):
    labels.append(f"Attn {i}")
    labels.append(f"Mlp {i}")

imshow(
    acts[:, :30],
    y=labels,
)



# %%
