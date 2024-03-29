# %%
from setup import *
from generate_activations import model, all_tokens, cfg

# %%
display(
    cv.tokens.colored_tokens(
        tokens=["hello", "there"], values=[0.001, 0.2], min_value=0
    )
)

# %%
all_tokens.shape

# %%
model.tokenizer.decode(all_tokens[1])

# %%
tokens = all_tokens[0]

_, cache = model.run_with_cache(
    tokens, stop_at_layer=cfg["layer"] + 1, names_filter=cfg["act_name"]
)

# %%
cache[cfg["act_name"]].shape

# %%
2 * (all_tokens.shape[0] / 32) / (3600)

# %%
