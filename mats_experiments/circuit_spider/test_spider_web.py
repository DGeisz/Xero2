# %%
%load_ext autoreload
%autoreload 2

# %%
from circuit_spider import CircuitSpider
from memory import get_gpu_memory
from plotly_utils import *
from pprint import pprint
import torch

from torch import tensor

import numpy as np

import einops


# %%

spider = CircuitSpider()


# %%
web = spider.create_prompt_web("14. Colorado 15. Michigan 16. Missouri 17")

# %%
web.test_compare_attn_out(11, -3)


# %%
web.tokens[0][-2]


# %%
(web.cache['embed'][0, -2] - web.model.W_E[web.tokens[0][-2]]).abs().max()

# %%

(web.cache['pos_embed'][0, -2] - web.model.pos_embed.W_pos[web.n_tokens - 2]).abs().max()



# %%
active_features = web.get_active_features(-2)
main_vector = active_features.scaled_vectors.sum(0)

# %%
web.cache['resid_post', 11].shape


# %%
torch.allclose(
    web.cache['resid_post', 11][0, -2]
    , main_vector), (web.cache['resid_post', 11][0, -2] - main_vector).abs().max()

# %%
# web = spider.create_prompt_web("Today I went to the store and I found that there was a very large animal that I found there")

# %%
web.cache['q', 0].shape

# %%
web.model.W_K[0].shape

# %%
web.cache['ln_final.hook_scale'].shape

# %%
web.model.W_U.shape


# %%
ll = einops.einsum(web.cache['ln_final.hook_normalized'], web.model.W_U, "b seq d_model, d_model dict -> b seq dict") + web.model.b_U

# %%
web.model.b_U.shape


# %%
(ll - web.logits).abs().max()
torch.allclose(ll, web.logits)




# %%
web.model.to_tokens(" 17")

# %%
active_features = web.get_active_features(-2)
main_vector = active_features.vectors.sum(0)

# %%
torch.allclose(web.cache['resid_post', 11][0, -2], main_vector), (web.cache['resid_post', 11][0, -2] - main_vector).abs().max()

# %%
active_features.get_total_active_features()

# %%
active_features.vectors.size(0)

# %%
active_features.values.shape






# %%



torch.allclose(web.cache['scale'], web.cache['ln_final.hook_scale'])

# %%
web.logits[0, -2]
# %%
token_i=1596
scale = web.cache['scale'][0, -1]

web.logits[0, -2, token_i], web.model.b_U[token_i]

# %%
(web.model.ln_final(main_vector) @ web.model.W_U + web.model.b_U)[token_i]

# %%
((main_vector / scale) @ web.model.W_U + web.model.b_U)[token_i]

# %%
torch.allclose(web.cache['resid_pre', 11], web.cache['resid_post', 10])

# %%
torch.allclose(web.model.ln_final(web.cache['resid_post', 11][0, -2]), web.cache['normalized'][0, -2])

# %%
(web.cache['normalized'][0, -2] @ web.model.W_U)[token_i]

# %%
main_vector.mean()

# %%
web.logits[0, -2, token_i], web.model.b_U[token_i]

# %%
web.z_saes[0].b_dec.shape

# %%
a = [1, 2]

# %%
a.extend([3, 4])

# %%
a

# %%
c, d, *_ = a

# %%
c, d

# %%
tensor(1) - 1











# %%
list(web.cache.keys())



# %%
main_vector.std()

# %%
web.model.ln_final


# %%
unembed_runs = web.get_unembed_lens_for_prompt_token(-2)

# %%
unembed_runs[2].run_data

# %%
web.get_active_features(-2)



# %%
web.logits.shape

# %%
unembed_runs

# %%
attn8 = unembed_runs[1]()

# %%
attn8[0]('q')



# %%
attn9 = unembed_runs[0]()

# %%

attn9_q = attn9[0](head_type='q')

# %%
a7_11 = attn9_q[0]()

# %%
a7_11[0]

# %%
attn9_q[0]



# %%

a7_11_k = a7_11[0]('v')

# %%
a7_11_k[1]()







# %%
vis, labels, acts = web.get_unembed_lens_for_prompt_token(-2, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)

# %%
feature_act = web.get_head_seq_activations_for_z_feature(9, -2, 18)
imshow(feature_act)

# %%
vis, labels, acts = web.get_q_lens_on_head_seq(9, 1, 7, 9, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)

# %%
vis, labels, acts = web.get_v_lens_at_seq(18, 9, 1, 7, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)

# %%
vis, labels, acts = web.get_v_lens_at_seq(38244, 7, 11, 7, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)

# %%
vis, labels, acts = web.get_k_lens_on_head_seq(7, 11, 7, 9, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)

# %%
vis, labels, acts = web.get_k_lens_on_head_seq(9, 1, 7, 9, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)

# %%
# feature_act = web.get_head_seq_activations_for_z_feature(4, 7, 14561)
# feature_act = web.get_head_seq_activations_for_z_feature(4, 7, 14561)
feature_act = web.get_head_seq_activations_for_z_feature(6, 7, 24355)
# feature_act = web.get_head_seq_activations_for_z_feature(7, 9, 38244)

# feature_act = web.get_head_seq_activations_for_z_feature(4, 7, 14561)
# feature_act = web.get_head_seq_activations_for_z_feature(5, -2, 23485)
imshow(feature_act)

# %%
vis, labels, acts = web.get_mlp_feature_lens_at_seq(8, 836, 7, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)





# %%
vis, labels, acts = web.get_unembed_lens_for_prompt_token(-2, visualize=True, k=10)
imshow(vis[:, :30], y=web.get_imshow_labels())
pprint(labels)

# %%







# %%
web.n_tokens

# %%
web.tokens, web.model.to_str_tokens(web.tokens)


# %%
# vis, labels, acts  = web.get_sae_feature_lens_on_head_seq(
#     1596, 
#     9,
#     1,
#     7,
#     visualize=True,
#     k=10
# )

vis, labels, acts  = web.get_k_lens_on_head_seq(
    9,
    1,
    7,
    9,
    visualize=True,
    k=10
)

# %%
acts.sum()

# %%
f = web.get_active_features( -2)
# %%
f.keys

# %%

web.cache['scale', 9, 'ln1']

# %%
web.cache['ln_final.hook_scale'][:, -3, :]

# %%
web.model.W_Q[0].shape
# %%
web.cache['k', 9].shape





# %%


# %%
vis.vectors.shape

# %%
spider.model.W_U.shape

# %%
ll = []

for i in range(12):
    ll.append(f"Attn {i}")
    ll.append(f"Mlp {i}")

imshow(
    vis[:, :30],
    y=ll,
)

# %%
labels
