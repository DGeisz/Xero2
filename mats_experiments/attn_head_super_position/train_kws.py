# %%
%load_ext autoreload
%autoreload 2

# %%
from attribution_buffer import AttributionBuffer
from kws import KWinnerSynthesis, KWSConfig
from plotly_utils import imshow
from data import select_token_range
from IPython.display import display
from transformer_lens import utils

import einops
import torch
import plotly.express as px
import circuitsvis as cv


# %%

lr = 0.01

simple_config = KWSConfig(
    attr_type="bos",
    num_features=20_000,
    n_winners=1,
    num_batches=100,
    update_mini_batch_size=200,
    log_freq=5,

    update_dead_neurons_freq=None,
    dead_neuron_resample_fraction=0.5,
    dead_neuron_resample_initial_delay=40,
    lr=lr,
    lr_schedule=[(20, lr), (100, lr * 4)]
)


# %%
buffer_mask = AttributionBuffer(simple_config.attr_type, 4096, normalize=True, mask=True)


# %%
kws = KWinnerSynthesis(simple_config, buffer=buffer_mask)

# %%
kws.train(1000)

# %%
torch.save(kws.features, 'interp_features.pt')

# %%


# kws.features[bigg[501]][:96].abs().sum()




# %%
tokens = select_token_range(0, 2000)
seq_attr = kws.get_sequence_attribution(N=2000)

# %%

top_heavy = (kws.features[:, 96:].abs().sum(dim=-1) > 2).nonzero().squeeze(-1).tolist()

top_heavy = sorted(top_heavy, key=lambda x: len(seq_attr[x]), reverse=True)

seq_attr_top = []

attr_lens = torch.tensor([len(a) for a in seq_attr])
bigg = attr_lens.float().argsort(descending=True)#[144:]

# %%
bigg = top_heavy
bigg_i = 13


imshow(einops.rearrange(kws.features[bigg[bigg_i]], "(l h) -> l h", l=12).cpu().float())

print("Feature:", bigg[bigg_i])
print("Top heavy score", kws.features[bigg[bigg_i]][96:].abs().sum().item())
print()

final = sorted(seq_attr[bigg[bigg_i]], key=lambda x: -x[2])
print(bigg_i, bigg[:5], len(final), len(seq_attr[bigg[bigg_i]]))


start = 10
amount = 10
# for i in range(10):
for batch, pos, value in final[start:start+amount]:
    # batch, pos, value = final[i]

    toks = tokens[batch]
    values=[0 for _ in range(len(toks))]
    values[pos + 1] = value

    toks = toks[pos-10:pos+10]
    values = values[pos-10:pos+10]

    display(cv.tokens.colored_tokens(tokens=kws._buffer.model.to_str_tokens(
        toks,
        ),
        values=values

    ))

# %%
model = kws._buffer.model


# %%
utils.test_prompt("for himself. Gollum got past Sam and attacked the invisible Frodo, biting off his finger, and finally regained his 'precious'. As he danced around in elation,", " G", model, prepend_bos=True)


# %%
i = 13

batch, pos, value = final[i]

toks = tokens[batch]
values=[0 for _ in range(len(toks))]
values[pos + 1] = value

# toks = toks[pos-10:pos+10]
# values = values[pos-10:pos+10]

display(cv.tokens.colored_tokens(tokens=kws._buffer.model.to_str_tokens(
    toks,
    ),
    values=values
))
    

# %%

print(bigg_i, bigg[:5], len(seq_attr[bigg[bigg_i]]), attr_lens[bigg[bigg_i]], len(sorted(seq_attr[bigg[bigg_i]], key=lambda x: -x[2])))

# %%
(kws.features[:, 96:].abs().sum(dim=-1) > 3).nonzero().squeeze(-1).shape



# %%
attr_lens.shape




# %%
batch, pos, value = final[2]

toks = tokens[batch]
values=[0 for _ in range(len(toks))]
values[pos] = value

# toks = toks[pos-10:pos+10]
# values = values[pos-10:pos+10]

display(cv.tokens.colored_tokens(tokens=kws._buffer.model.to_str_tokens(
    toks,
    ),
    values=values
))




# %%


# %%
attr_lens = torch.tensor([len(a) for a in seq_attr])

# %%
attr_lens.float().max()
# %%

attr_lens.float().argsort(descending=True)#[144:]



# %%
imshow(einops.rearrange(kws.features[bigg[bigg_i]], "(l h) -> l h", l=12).cpu().float())

# %%
sorted(seq_attr[6711], key=lambda x: -x[2])






# %%
def reshape(data, l=12):
    return einops.rearrange(data.to(torch.float32), "(l h) -> l h", l=l)


# %%
out = kws.run_and_reconstruct_single_batch(include_winner_features=True)
sampler = iter(range(80, 100))


# %%
bigg_i = next(sampler)


d = reshape(out.data[bigg_i])
r = reshape(out.reconstructed_data[bigg_i])
diff = d - r

barrier = torch.ones(12, 1).cuda() * d.max() / 2

all_stuff = [
    d,
    r,
    diff
]

imshow(torch.stack(all_stuff).cpu(), facet_col=0)

fig = px.imshow(einops.rearrange(out.winner_features[bigg_i], "n (l h) -> n l h", l=12).cpu().to(torch.float32), 
        facet_col=0, 
        # facet_col_wrap=20, 
        facet_col_wrap = 5,
        facet_row_spacing = 0.001,
        facet_col_spacing = 0.01,
        # height = 5000,
        color_continuous_midpoint=0.0,
        labels={"facet_col": ""},

        color_continuous_scale="RdBu",
        width=800
            )

fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

fig.show()


# %%
