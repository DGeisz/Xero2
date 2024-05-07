# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import einops

from attribution_buffer import load_attribution_tensor_locally
from data import select_token_range
from attention_attribution import get_attn_attrib_on_seq, get_bos_ablate_for_head
from transformer_lens import HookedTransformer

from plotly_utils import imshow

from d_types import DTYPES

device = "cuda:0"


# %%
cfg = {"model": "gpt2", "device": device, "enc_dtype": "bf16"}

model = (
    HookedTransformer.from_pretrained(cfg["model"])
    .to(DTYPES[cfg["enc_dtype"]])
    .to(cfg["device"])
)


# %%
tokens = select_token_range(0, 100).to(cfg["device"])
bos_ablate_for_head = get_bos_ablate_for_head(model, tokens)

# %%

# %%
data = load_attribution_tensor_locally("bos", 0, 1000)

# %%
data.shape

# %%
new_data = einops.rearrange(data, "(batch seq_len) h l -> batch seq_len h l", seq_len=119)

# %%
new_data.shape

# %%
bos.shape

# %%
torch.allclose(new_data[0], bos.cpu())


# %%
(new_data[3] - bos.cpu()).abs().mean()

# %%
ti = 10

bos, zero, mix = get_attn_attrib_on_seq(
    model, tokens[ti].unsqueeze(0), 10, bos_ablate_for_head
)

# %%
i = 20

imshow(torch.stack([new_data[ti][i], bos[i].cpu()]).float(), facet_col=0)

# %%
