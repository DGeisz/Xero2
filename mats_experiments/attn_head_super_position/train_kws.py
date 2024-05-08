# %%
%load_ext autoreload
%autoreload 2

# %%
from attribution_buffer import AttributionBuffer
from kws import KWinnerSynthesis, KWSConfig
from plotly_utils import imshow
from data import select_token_range
from IPython.display import display, HTML
from transformer_lens import utils
from typing import Union, Optional, List
from jaxtyping import Float
from transformer_lens import ActivationCache

import einops
import torch
import plotly.express as px
import circuitsvis as cv

# %%
def h(layer: int, head: int):
    return (12 * layer) + head


def visualize_attention_patterns(
    heads: Union[List[int], int, Float[torch.Tensor, "heads"]],
    local_cache: ActivationCache,
    local_tokens: torch.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[torch.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][batch_index, head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = model.to_str_tokens(local_tokens)

    # Combine the patterns into a single tensor
    patterns: Float[torch.Tensor, "head_index dest_pos src_pos"] = torch.stack(
        patterns, dim=0
    )

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = cv.attention.attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"



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
    # lr_schedule=[(20, lr), (100, lr * 4), (200, lr / 10)],
    # mask_initial_neurons=True
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
bos, winners = kws.get_features_for_string("14. Michigan 15. Colorado 16. California 17.")

# %%
imshow(bos[10].cpu().float())

# %%
i = 11

print("Winner:", winners[i].item())
imshow(einops.rearrange(kws.features[winners[i].item()], "(l h) -> l h", l=12).cpu().float())


# %%
winners[10]






# %%

tokens = select_token_range(0, 8000)
seq_attr = kws.get_sequence_attribution(N=8000)

# %%
start = 96
amount = 48

thres = 2


top_heavy = (kws.features[:, start:start+amount].abs().sum(dim=-1) > thres).nonzero().squeeze(-1).tolist()

top_heavy = sorted(top_heavy, key=lambda x: len(seq_attr[x]), reverse=True)
print("num top heavy:", len(top_heavy))

attr_lens = torch.tensor([len(a) for a in seq_attr])



def num_to_str(num):
    layer = num // 12
    head = num % 12

    return f"L{layer}H{head}"

def print_top_heads(feature, thresh=.3):
    heads = (kws.features[feature].abs() > thresh).nonzero().squeeze(1).tolist()

    print(", ".join([num_to_str(h) for h in heads]))

# %%
bigg = attr_lens.float().argsort(descending=True)#[144:]

# %%

(kws.features[0].abs() > .1).nonzero().squeeze(1).tolist()

# %%
top = kws.get_top_feature_for_heads([
    # (10, 10, 1), 
    # (8, 10, 1),
    # (8, 2, -1),
    # (11, 2, 1),
    (0, 9, 1),
    (9, 5, 1),
    # (10, 7, 1),
    # (11, 10, 1)
    (11, 0, 1)

    # (11, 4, 1)
    ])

top[:5]




# %%
bigg = top_heavy
# bigg = top

bigg_i = 22

feature_i = bigg[bigg_i]
# feature_i = 15678

start = 0
amount = 10

imshow(
    einops.rearrange(kws.features[feature_i], "(l h) -> l h", l=12).cpu().float(),
    title=f"Circuit Pattern #{feature_i}",
labels={"x": "Head", "y": "Layer"}
)

print("Feature:", feature_i)
print("Top heavy score", kws.features[feature_i][96:].abs().sum().item())
print_top_heads(feature_i, .1)
print()

final = sorted(seq_attr[feature_i], key=lambda x: -x[2])
print(bigg_i, bigg[:5], len(final), len(seq_attr[feature_i]))

# for i in range(10):
# for batch, pos, value in seq_attr[bigg[bigg_i]][start:start+amount]:
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
        values=values, 
        positive_color='blue'
    ))

# %%
(kws._buffer.bos_ablate_for_head == True).sum() / 144


# %%
data = kws._buffer.next()

# %%
imshow(
    einops.rearrange(data[21], "(l h) -> l h", l=12).cpu().float(),
    title=f"Head Attribution Pattern",
    labels={"x": "Head", "y": "Layer"}
)




# %%
i = 0
pos_amount = 20

batch, pos, value = final[i]

toks = tokens[batch]
values=[0 for _ in range(len(toks))]
values[pos + 1] = value

print(model.tokenizer.decode(toks[pos-pos_amount:pos+pos_amount]))

display(cv.tokens.colored_tokens(tokens=kws._buffer.model.to_str_tokens(
    toks,
    ),
    values=values
))

# %%
seq = """
 of Neue Zeit (New Times), organ of the German Social-Democrats, in July 1898.

[2]
"""
# seq = "the Pew Research Center finds that there is a correlation between church goers and attitudes toward torture"
new_tokens = model.to_tokens(seq)

_, cache = model.run_with_cache(new_tokens)

HTML(visualize_attention_patterns(
    [h(9, 5), h(11, 0)],
    cache,
    local_tokens=model.to_tokens(seq),
))
    

# %%
model = kws._buffer.model


# %%
utils.test_prompt("for himself. Gollum got past Sam and attacked the invisible Frodo, biting off his finger, and finally regained his 'precious'. As he danced around in elation,", " G", model, prepend_bos=True)

# %%
bigg[bigg_i]

# %%
seq_attr[bigg[bigg_i]]

# %%
tokens = select_token_range(0, 2000)

# %%
final




# %%
model.tokenizer.decode(tokens[1293][50:])





# %%
i = 0

batch, pos, value = final[i]

toks = tokens[batch]
values=[0 for _ in range(len(toks))]
values[pos + 1] = value

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
seq = "the Pew Research Center finds that there is a correlation between church goers and attitudes toward torture"
tokens = model.to_tokens(seq)

_, cache = model.run_with_cache(tokens)

HTML(visualize_attention_patterns(
    [h(9, 5), h(8, 10)],
    cache,
    local_tokens=model.to_tokens(seq),
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
