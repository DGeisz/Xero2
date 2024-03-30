# %%
%load_ext autoreload
%autoreload 2


# %%
import torch
import torch as t
import einops
import tqdm
import time

from k_winner_synthesis import KWinnerSynthesis, KWSConfig, cfg as buffer_config
from generate_activations import model, all_tokens, cfg
from setup import *


# %%
simple_config = KWSConfig(
    num_features=4_000,
    n_winners=5,
    num_batches=100,
    lr=0.005,
    log_freq=5,
    update_dead_neurons_freq=None,
    dead_neuron_resample_fraction=0.1,
    dead_neuron_resample_initial_delay=50,
    lr_schedule=[(20, 0.005), (100, 0.01)],
)

kws = KWinnerSynthesis(config=simple_config, buffer_config=buffer_config)

kws.train(1000)

# %%
kws.config.log_freq = 20

kws.train(1000)


# %%


# %%
model_batch_size = 32

tokens = all_tokens[:model_batch_size]

# %%
tokens.shape

# %%
_, cache = model.run_with_cache(
    tokens, stop_at_layer=cfg["layer"] + 1, names_filter=cfg["act_name"]
)

acts = cache[cfg["act_name"]]


# %%
data = einops.rearrange(acts, "n s d -> (n s) d").to(torch.bfloat16)
# winners = einops.rearrange(
#     kws.get_winners(data), "(n s) w -> n s w", n=model_batch_size
# )

# %%
a, v = kws.get_winners(data)


# %%


start = time.time()

model_batch_size = 32 * 2 * 2 * 2


num_seqs = model_batch_size
# num_seqs = model_batch_size * 2 * 2 * 2
winner_i = [
    t.tensor([]).to(device) for _ in range(kws.config.num_features)
]

seq_len = 128

for indices in torch.split(t.arange(num_seqs), model_batch_size):
    tokens = all_tokens[indices]

    _, cache = model.run_with_cache(
        tokens, stop_at_layer=cfg["layer"] + 1, names_filter=cfg["act_name"]
    )

    acts = cache[cfg["act_name"]]

    data = einops.rearrange(acts, "n s d -> (n s) d").to(torch.bfloat16)

    winner_indices, winner_values = kws.get_winners(data, n_winners=5)

    winner_indices = einops.rearrange(
        winner_indices, "(n s) w -> n s w", n=model_batch_size
    )
    winner_values = einops.rearrange(
        winner_values, "(n s) w -> n s w", n=model_batch_size
    )

    seq_index_start = indices[0].item()

    for f in tqdm(range(kws.config.num_features)):
        win_tensor_locs = t.where(winner_indices == f)

        win_locs = t.stack(
            [win_tensor_locs[0], win_tensor_locs[1], winner_values[win_tensor_locs]],
            dim=0,
        ).T + t.tensor([[seq_index_start, 0, 0]]).to(device)

        winner_i[f] = t.cat([winner_i[f], win_locs], dim=0)


print(f"Took {time.time() - start:.2f} seconds")

# %%
all_tokens.shape

# %%
sum(winner_i[i].shape[0] for i in range(kws.config.num_features))


# %%
def run_for_feature(i, start, q):
    seq_dict = {}

    feature_winners = winner_i[i]

    for i, k, l in feature_winners.tolist():
        if i in seq_dict:
            seq_dict[i].append((k, l))
        else:
            seq_dict[i] = [(k, l)]


    _, indices = torch.sort(feature_winners[:, 2], descending=True)
    sorted_winners = feature_winners[indices]

    all_max = []
    display_max = 20

    for i, _, _ in sorted_winners:
        index = int(i)

        if index not in all_max:
            all_max.append(index)

        if len(all_max) >= display_max:
            break

    print(all_max)

    for seq_i in all_max[start:start+q]:
        # seq_i = int(list(seq_dict.keys())[key_i])
        tt = model.to_str_tokens(all_tokens[seq_i])

        v = [0.0 for _ in range(128)]

        seq = seq_dict[seq_i]

        for i, k in seq:
            v[int(i)] = k

        indices = [i for i, _ in seq]

        padding = 10

        start = max(int(min(indices)) - padding, 0)
        end = int(max(indices)) + padding
        
        tokens = tt[start:end]
        values = v[start:end]

        # tokens = tt
        # values = v

        tokens.extend(["\n      ", ' '])
        values.extend([0, 0])

        
        display(cv.tokens.colored_tokens(tokens=tokens, values=values, min_value=0.1, max_value=1))


def most_similar(i):
    dots = einops.einsum(kws.features[i], kws.features, "d, f d -> f")
    return dots.sort(descending=True).indices[:10].tolist()



# %%
run_for_feature(1, 10, 10)

# %%
most_similar(3000)

# %%
winner_count = kws.get_winner_count_from_n_batches(10)

# %%
winner_count.argmax()
# %%
winner_count.mean()

# %%
model.unembed.W_U.dtype

# %%
logits = einops.einsum(kws.features[0], model.unembed.W_U.to(torch.bfloat16), 'f, f d -> d')

values, indices = logits.sort(descending=True)

print("Most likely next:",
model.to_str_tokens(indices[:10])
      )

print("Least likely next:",
model.to_str_tokens(indices[-10:])
      )




# %%
winner_i[0].shape

_, indices = torch.sort(winner_i[0][:, 2], descending=True)

# Use the indices to sort the entire tensor
sorted_tensor = winner_i[0][indices]

# winner_i[200]
# %%
sorted_tensor

# %%

# %%
all_max




# %%
kws.features.norm(dim=-1).std()


# %%

for i in winners[0, 0]:
    print(i.item())

# %%

# %%
list(enumerate(winners[0, 0]))

# %%
winners.shape

# %%
winner_i[0]


# %%
wt = t.tensor([len(d) for d in winner_i])

# %%
wt.float().sum()


# %%
wt.shape

# %%
ww = t.where(winners == 0)

# %%
a = []
# %%


bb = (t.stack([ww[0], ww[1]], dim=0).T + t.tensor([[1, 0]]).to(device)).tolist()

bb = t.tensor(bb)
bb

# %%
bb = t.cat([bb, bb], dim=0)

# %%
for i in range(100):
    print("\n\n SEQ", i)
    print(model.tokenizer.decode(all_tokens[i]))


# %%
a.extend(bb)
# %%
a


# %%
list((q.unsqueeze(0) for q in ww))

# %%
wt[0]

# %%
winners[21, 38, 4]

# %%
winner_i[300]

# %%
ii = [
    j[1]
    for j in [
        # (193, 2),
        # (193, 7),
        # (193, 14),
        # (193, 18),
        # (193, 25),
        # (193, 30),
        # (193, 32),
        # (193, 34),
        # (193, 44),
        # (193, 50),
        # (193, 57),
        # (193, 73),
        # (193, 87),
        # (193, 95),
        # (233, 11),
        # (233, 12),
        # (233, 33),
        # (233, 34),
        # (233, 47),
        # (233, 48),
        # (233, 86),
        # (445, 91),
        # (445, 126),
        #  [499, 99],
        #  [499, 100],
        #  [376, 25],
        #  [376, 41],
        #  [376, 51],
        #  [376, 74],
        #  [387, 3],
        #  [387, 49],
        #  [387, 60],
        #  [387, 86],
        [357, 10],
        [357, 76],
        [357, 85],
        [357, 105],
        [357, 125],
        # (247, 3),
        # (247, 11),
        # (247, 66),
        # (247, 71),
        # (247, 79),
        # (247, 83),
        # (247, 86),
        # (247, 99),
        # (247, 100),
        # (247, 111),
        # (247, 113),
        # (247, 124),
    ]
]

tt = model.to_str_tokens(all_tokens[357])

v = [0 for _ in range(128)]

for i in ii:
    v[i] = 1

cv.tokens.colored_tokens(tokens=tt, values=v)

# %%
winner_count = kws.get_winner_count_from_n_batches(30)

# %%
winner_count.argmax()

# %%
winner_i[360]

# %%
data.shape

# %%
data = data / data.norm(dim=-1).unsqueeze(-1)

raw_output = einops.einsum(data, kws.features, "n d, f d -> n f")

sorted_output = t.sort(raw_output, descending=True, dim=-1)

winners = sorted_output.indices[:, : kws.config.n_winners]
win_vals = sorted_output.values[:, : kws.config.n_winners]

# %%
win_vals


# %%
win_vals[20]

# %%
data.norm(dim=-1).min()



# %%

# %%
winners.shape

# %%
raw_output.shape

# %%

# %%
vals = t.sort(raw_output, descending=True, dim=-1)
# 

# %%
vals

# %%
winners

# %%
winner_indices, winner_values = kws.get_winners(data)

winner_indices = einops.rearrange(
    winner_indices, "(n s) w -> n s w", n=model_batch_size
)
winner_values = einops.rearrange(
    winner_values, "(n s) w -> n s w", n=model_batch_size
)

# %%
win_tensor_locs = t.where(winner_indices == 0)

# %%
t.stack(
    [win_tensor_locs[0], win_tensor_locs[1], winner_values[win_tensor_locs]],
    dim=1
)

# %%
winner_values[win_tensor_locs].shape

# %%
winner_values[2, 50, 5]

# %%
