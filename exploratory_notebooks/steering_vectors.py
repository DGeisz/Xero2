# %%
%load_ext autoreload
%autoreload 2


# %%
import torch
import transformer_lens
import sae_vis

from typing import *
from transformer_lens import HookedSAETransformer, HookedSAE
from datasets import load_dataset
from IPython.display import HTML, display

# %%
torch.set_grad_enabled(False)

# %%
model = HookedSAETransformer.from_pretrained_no_processing("gpt2-xl")

# %%
gpt2xl = model


# %%
sae = HookedSAE.from_pretrained("gpt2-xl-saex-resid-pre-l20")

# %%

# %%
data = load_dataset("Elriggs/openwebtext-100k")
data = data['train']

# %%
data[0]

# %%

# %%
SEQ_LEN = 128

# Tokenize the data (using a utils function) and shuffle it
tokenized_data = transformer_lens.utils.tokenize_and_concatenate(data, gpt2xl.tokenizer, max_length=SEQ_LEN) # type: ignore
tokenized_data = tokenized_data.shuffle(42)

# Get the tokens as a tensor
all_tokens = tokenized_data["tokens"]
assert isinstance(all_tokens, torch.Tensor)

print(all_tokens.shape)

# %%
sae_vis_sae_cfg = sae_vis.model_fns.AutoEncoderConfig(
    d_in=sae.cfg.d_in,
    d_hidden=sae.cfg.d_sae
)

# %%
sae_vis_sae = sae_vis.model_fns.AutoEncoder(sae_vis_sae_cfg).to("cuda:0")
sae_vis_sae.load_state_dict(sae.state_dict())

# %%
site = "blocks.20.hook_resid_pre"


