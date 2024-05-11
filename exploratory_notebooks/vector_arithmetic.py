# %%
%load_ext autoreload
%autoreload 2

# %%
from transformer_lens import HookedTransformer
from typing import List, Tuple

import einops
import torch

# %%
torch.set_grad_enabled(False)


# %%
model = HookedTransformer.from_pretrained("pythia-70m")


# %%
model.to_tokens(" King", prepend_bos=False)

# %%
model.tokenizer.encode("King", prepend_bos=False)




# %%
def avg_vector(tokens: List[str]):
    vecs = [model.W_E[model.tokenizer.encode(token)[1]] for token in tokens]
    vecs = [vec / vec.norm() for vec in vecs]
    
    return sum(vecs) / len(vecs)

def most_likely_token_for_vector(vec):
    max_token = einops.einsum(vec, model.W_E, "d, f d -> f").argmax().item()
    return model.tokenizer.decode([max_token])




# %%
def vector_arithmetic(tokens: List[Tuple[str, int]]):
    vecs = [(model.W_E[model.tokenizer.encode(token)[1]], weight) for token, weight in tokens]
    vecs = [(vec / vec.norm(), weight) for vec, weight in vecs]
    
    vec = sum(weight * vec for vec, weight in vecs)

    max_token = einops.einsum(vec, model.W_E, "d, f d -> f").argmax().item()

    return model.tokenizer.decode([max_token])

# %%
vector_arithmetic([
    (" King", 1),
    (" Man", -1),
    (" Woman", 1)
])



# %%
king = model.W_E[model.tokenizer.encode("King")[1]]
man = model.W_E[model.tokenizer.encode("Man")[1]]
woman = model.W_E[model.tokenizer.encode("Woman")[1]]

king /= king.norm()
man /= man.norm()
woman /= woman.norm()

# %%
possible_queen = king - man + woman
# possible_queen = man


# %%
max_token = einops.einsum(possible_queen, model.W_E, "d, f d -> f").argmax().item()
model.tokenizer.decode([max_token])

# %%
week_vector = avg_vector([
    " Sunday",
    " Monday",
    " Tuesday",
    " Wednesday",
    " Thursday",
    " Friday",
    " Saturday"
])

# %%
month_vector = avg_vector([
    " January",
    " February",
    " March",
    " April",
    " May",
    " June",
    " July",
    " August",
    " September",
    " October",
    " November",
    " December"
])

# %%
wednesday = model.W_E[model.tokenizer.encode(" Wednesday")[0]]
wednesday /= wednesday.norm()

idk = wednesday - week_vector + month_vector


# %%
most_likely_token_for_vector(month_vector)

# %%
