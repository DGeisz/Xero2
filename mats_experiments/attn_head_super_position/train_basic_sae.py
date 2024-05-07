# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import tqdm
import plotly.express as px
import einops

from attribution_buffer import AttributionBuffer
from basic_sae import AttributionSAE
from pprint import pprint
from transformer_lens import utils


# %%
def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()




# %%
cfg = {
    "batch_size": 4096,
    "n_features": 10_000,

    "lr": 1e-3,
    # "l1_coeff": 0,
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
}

# %%
buffer = AttributionBuffer("mix", cfg['batch_size'])

# %%
encoder = AttributionSAE(attr_type="mix", n_features=cfg['n_features'], l1_coeff=cfg['l1_coeff'])

def get_dead_features(buffer, encoder, sample_batches=4):
    all_acts = []

    for _ in range(sample_batches):
        data = buffer.next().to(torch.float32)
        all_acts.append(encoder(data)[2])

    all_acts = torch.cat(all_acts, dim=0)

    return (all_acts.sum(dim=0) == 0).sum().item()

# %%
num_batches = 1000

# model_num_batches = cfg["model_batch_size"] * num_batches
encoder_optim = torch.optim.Adam(
    encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
)

recons_scores = []
act_freq_scores_list = []
for i in range(num_batches):
    attn_attr = buffer.next().to(torch.float32)

    attn_attr /= attn_attr.norm(dim=-1, keepdim=True)


    loss, x_reconstruct, mid_acts, l2_loss, l1_loss, l0 = encoder(attn_attr)
    loss.backward()

    encoder.make_decoder_weights_and_grad_unit_norm()
    encoder_optim.step()
    encoder_optim.zero_grad()

    loss_dict = {
        "loss": loss.item(),
        "l2_loss": l2_loss.item(),
        "l1_loss": l1_loss.item(),
        "l0": l0.item(),
    }

    if (i) % 100 == 0:
        print("Batch:", i)
        print("Loss:", loss.item(), "L2:", l2_loss.item(), "L1:", l1_loss.item(), "L0:", l0.item())
        print("Dead Features: ", get_dead_features(buffer, encoder))
        # pprint(loss_dict)
        print()

    del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, attn_attr


# %%
def reshape(data):
    return einops.rearrange(data, "(l h) -> l h", l=12)


# %%
data = buffer.next()
data /= data.norm(dim=-1, keepdim=True)

re = encoder(data)[1]

# %%
k = 100

all_ims = []
re_ims = []
for i in range(4):
    all_ims.append(reshape(data[k + i]).to(torch.float32))
    all_ims.append(torch.ones((12, 1)).to(torch.float32).cuda() / 100)
    re_ims.append(reshape(re[k + i]).to(torch.float32))
    re_ims.append(torch.ones((12, 1)).to(torch.float32).cuda() / 100)

ai = torch.cat(all_ims, dim=1)
ri = torch.cat(re_ims, dim=1)

imshow(ai)
imshow(ai - ri)
# imshow(torch.concat(all_ims, dim=1))
# imshow(torch.concat(re_ims, dim=1))
# imshow(torch.concat(re_ims, dim=1))



# imshow(reshape(data[i]).to(torch.float32), width=400)
# imshow(reshape(data[i + 1]).to(torch.float32), width=400)
# imshow(reshape(data[i + 2]).to(torch.float32), width=400)
# imshow(reshape(re[0]).to(torch.float32))
# imshow(reshape(re[i]).to(torch.float32))
# imshow(reshape(data[i] - re[i]).to(torch.float32))



# %%
attn_attr = encoder(data)[2]

# %%
a = 0
b = 100

(re[a] - re[b]).abs().max()

# %%
re.shape

# %%
data.shape





# %%
(attn_attr.sum(dim=0) == 0).sum()

# %%


all_acts = []

for i in range(4):
    data = buffer.next().to(torch.float32)
    all_acts.append(encoder(data)[2])

all_acts = torch.cat(all_acts, dim=0)

# %%
all_acts.sum(dim=0).max()

# %%
