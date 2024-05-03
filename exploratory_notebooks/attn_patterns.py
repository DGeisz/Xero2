# %%
%load_ext autoreload
%autoreload 2

# %%
from setup import *
from generate_activations import *
from IPython.display import HTML

# %%
model = HookedTransformer.from_pretrained('gpt2').to(DTYPES[cfg["enc_dtype"]]).to(cfg["device"])


# %%
_, cache = model.run_with_cache(all_tokens[:10].to(cfg["device"]))

# %%
def plot_layer_heads(layer: int):
    pattern = einops.reduce(cache['pattern', layer], "b h x y -> h x y", 'mean')

    return cv.attention.attention_heads(pattern[:, :40, :40], tokens=model.to_str_tokens(all_tokens[0][:40])).show_code()

# %%
HTML(plot_layer_heads(2))


# %%
list(cache.keys())




# %%
len(model.blocks)

# %%
