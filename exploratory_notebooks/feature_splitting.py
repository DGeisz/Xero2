# %%
%load_ext autoreload
%autoreload 2

# %%
import wandb
import torch
import einops

from transformer_lens import HookedSAETransformer, HookedTransformer, HookedSAE, HookedSAEConfig
from sae_lens import LMSparseAutoencoderSessionloader
from pathlib import Path
from plotly_utils import *
from tqdm import trange

import torch.nn.functional as F
import matplotlib.pyplot as plt

# %%
from huggingface_hub import hf_hub_download


# %%
wandb.init()


# %%
run = wandb.init()
artifact = run.use_artifact(
    "jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_98304:v0",
    type="model",
)

artifact_dir = artifact.download()

# # %%
# !pip install sae_lens


# %%
REPO_ID = "jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8"
REPO_ID = "pchlenski/gpt2-transcoders"
# FILENAME = f"blocks.8.hook_resid_pre_768"
# FILE_NAME = "blocks.8.hook_resid_pre_12288"

path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME)

# %%
model, sparse_autoencoder, activation_store = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
    path = path
)

# %%
folder_dir = "/root/GPT2-Small-Feature-Splitting-Experiment-Layer-8/"


# %%

def folder_to_file(folder):
    folder = Path(folder)
    files = list(folder.glob("*"))
    files = [str(f) for f in files]
    return files[0] if len(files) == 1 else files


from sae_lens.training.utils import BackwardsCompatiblePickleClass

f = folder_to_file(artifact_dir)
blob = torch.load(f, pickle_module=BackwardsCompatiblePickleClass)
config_dict = blob["cfg"].__dict__
state_dict = blob["state_dict"]
# %%

cfg = HookedSAEConfig(
    d_sae=config_dict["d_sae"],
    d_in=config_dict["d_in"],
    hook_name=config_dict["hook_point"],
    use_error_term=False,
    dtype=torch.float32,
    seed=None,
    device="cuda",
)
print(cfg)
sae = HookedSAE(cfg)
sae.load_state_dict(state_dict)

# %%
saes = []
for n in range(8):
    artifact = run.use_artifact(
        f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{2**n * 768}:v0",
        type="model",
    )
    artifact_dir = artifact.download()
    f = folder_to_file(artifact_dir)
    blob = torch.load(f, pickle_module=BackwardsCompatiblePickleClass)
    config_dict = blob["cfg"].__dict__
    state_dict = blob["state_dict"]
    cfg = HookedSAEConfig(
        d_sae=config_dict["d_sae"],
        d_in=config_dict["d_in"],
        hook_name=config_dict["hook_point"],
        use_error_term=False,
        dtype=torch.float32,
        seed=None,
        device="cuda",
    )
    print(cfg)
    sae = HookedSAE(cfg)
    sae.load_state_dict(state_dict)
    saes.append(sae)
# %%
saes[1].W_dec[:, None, :].shape

# %%
torch.set_grad_enabled(False)


# %%
# start = 6

i = 7

s1 = saes[i].W_dec
s1 /= s1.norm(dim=-1, keepdim=True)

s2 = saes[i].W_dec
s2 /= s2.norm(dim=-1, keepdim=True)

cos_sim = einops.einsum(s1[:10_000], s2[:10_000], "d1 d_model, d2 d_model -> d1 d2")

# N = cos_sim.size(0)
cos_sim[range(N), range(N)] = 0

# cos_sim.cpu().max(dim=0)

histogram(
    cos_sim.max(dim=0).values.cpu(),
    title=f"Max Cosine for Bigger SAE features ({s1.shape[0]})"
)

histogram(cos_sim.max(dim=1).values.cpu(),
    title=f"Max Cosine for Smaller SAE features ({s2.shape[0]})"
)

del cos_sim



# %%
HookedSAE

# %%
s0 = saes[0].W_dec

# %%
s_small = saes[0]
s_big = saes[5]

# %%



# %%

s_big(s_small.W_dec).shape
s_big.hook_sae_acts_post.value

# %%
def get_acts(sae, x):
    x_cent = x - sae.b_dec
    # WARNING: if editing this block of code, also edit the error computation inside `if sae.cfg.use_error_term`
    sae_acts_pre = sae.hook_sae_acts_pre(
        einops.einsum(x_cent, sae.W_enc, "... d_in, d_in d_sae -> ... d_sae")
        + sae.b_enc  # [..., d_sae]
    )
    return F.relu(sae_acts_pre)

# %%
acts = get_acts(s_small, s_big.W_dec)

# %%
s_big = saes[6]

# %%
2 ** 8



# %%
num_compressed = 1
# num_compressed = 2 ** 8
print("Num Compressed:", num_compressed)

data = einops.rearrange(s_big.W_dec, "(a b) d_model -> a b d_model", a=num_compressed).sum(dim=0)
print(data.shape)

acts = get_acts(s_small, data)
acts_sum = acts.sum(dim=-1).float()

print()
print(f'Acts | Sum: {acts_sum.mean().item():.3g} | Std: {acts_sum.std().item():.3g} | Max: {acts_sum.max().item():.3g} | Min: {acts_sum.min().item():.3g}')
print()
re_data = s_small(data)
print()
print(re_data.shape)

print(f"Avg error: {(re_data - data).norm(dim=-1).mean().item():.4g}")

# %%
saes[6].W_dec.shape


# %%
N, _ = s_big.W_dec.shape

sae = saes[6]

summ = []
std = []
maxx = []
minn = []
error = []

d_error = []

for i in trange(20, 2000):
    num = (N // i) * i
    w_dec_eff = s_big.W_dec[:num]
    # print(w_dec_eff.shape)

    data = einops.rearrange(w_dec_eff, "(a b) d_model -> a b d_model", a=i).sum(dim=0)

    acts = get_acts(sae, data)
    acts_sum = acts.sum(dim=-1).float()

    summ.append(acts_sum.mean().item())
    std.append(acts_sum.std().item())
    maxx.append(acts_sum.max().item())
    minn.append(acts_sum.min().item())

    re_data = sae(data)

    error.append((re_data - data).norm(dim=-1).mean().item())

    de_data = sae(re_data)

    d_error.append((re_data - de_data).norm(dim=-1).mean().item())

# %%
line(error, height=300)
line(d_error, height=300)
line(summ, height=300)
    



# %%
#  line([1, 2])

plt.plot(summ)
plt.plot([a + b for a, b in zip(summ, std)])
plt.plot([a - b for a, b in zip(summ, std)])
plt.plot(maxx)
plt.plot(minn)

plt.show()
# %%
line(summ)






# %%

i = 2
num = (N // i) * i
w_dec_eff = s_big.W_dec[:num]
# print(w_dec_eff.shape)

data = einops.rearrange(w_dec_eff, "(a b) d_model -> a b d_model", a=i).sum(dim=0)
print(data.shape)

acts = get_acts(s_small, data)
acts_sum = acts.sum(dim=-1).float()

print(acts_sum.mean().item())
print(acts_sum.std().item())
print(acts_sum.max().item())
print(acts_sum.min().item())

re_data = s_small(data)

print((re_data - data).norm(dim=-1).mean().item())


# %%
re = s_small.forward(s_big.W_dec)

# %%
histogram((re - s_big.W_dec).norm(dim=-1).cpu())



# %%
(acts > 0).sum(dim=-1).float().mean()

# re = s_big(s_small.W_dec)

# %%
(re - s_small.W_dec)[0]

# %%
(acts > 0).sum(dim=-1)

# %%
w_small = saes[1].W_dec

# %%
i = 0

w_small = saes[i].W_dec
# w_small += saes[i].b_dec.unsqueeze(0)
acts = get_acts(saes[i], w_small)
N = acts.size(0)
acts[range(N), range(N)] = 0

imshow(acts)
# saes[1].b_dec.shape


# %%
w_small.shape

# %%

# %%


