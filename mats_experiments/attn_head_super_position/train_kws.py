# %%
%load_ext autoreload
%autoreload 2

# %%
from attribution_buffer import AttributionBuffer
from kws import KWinnerSynthesis, KWSConfig
from plotly_utils import imshow

import einops
import torch

# %%

lr = 0.01

simple_config = KWSConfig(
    num_features=40_000,
    n_winners=10,
    num_batches=100,
    update_mini_batch_size=200,
    log_freq=5,

    update_dead_neurons_freq=None,
    dead_neuron_resample_fraction=0.5,
    dead_neuron_resample_initial_delay=40,
    lr=lr,
    lr_schedule=[(20, lr), (100, lr * 4)]
)

# # %%
# buffer = AttributionBuffer('mix', 4096, normalize=True)

# %%
buffer_mask = AttributionBuffer('mix', 4096, normalize=True, mask=True)

# %%
kws = KWinnerSynthesis(simple_config, buffer=buffer_mask)

# %%
kws.train(1000)

# %%

# %%
def reshape(data, l=12):
    return einops.rearrange(data.to(torch.float32), "(l h) -> l h", l=l)


# %%
import plotly.express as px



# %%
fig = px.imshow(einops.rearrange(out.winner_features[0], "n (l h) -> n l h", l=12).cpu().to(torch.float32), 
        facet_col=0, 
        # facet_col_wrap=20, 
        facet_col_wrap = 10,
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
out = kws.run_and_reconstruct_single_batch(include_winner_features=True)

# %%
a = iter(range(80, 100))


# %%
i = next(a)


d = reshape(out.data[i])
r = reshape(out.reconstructed_data[i])
diff = d - r

barrier = torch.ones(12, 1).cuda() * d.max() / 2

all_stuff = [
    d,
    r,
    diff
]

imshow(torch.stack(all_stuff).cpu(), facet_col=0)

fig = px.imshow(einops.rearrange(out.winner_features[i], "n (l h) -> n l h", l=12).cpu().to(torch.float32), 
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

# %%
winner_features.shape



# %%
wi[0]




# %%

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

# Sample data: list of 12x12 matrices
data = [np.random.rand(12, 12) for _ in range(4)]

# Create a subplot grid: 2 rows, 2 columns
fig = make_subplots(rows=2, cols=2, subplot_titles=["Heatmap 1", "Heatmap 2", "Heatmap 3", "Heatmap 4"])

# Fill subplots
for i, matrix in enumerate(data):
    row = i // 2 + 1
    col = i % 2 + 1
    fig.add_trace(
        px.imshow(matrix),

        # go.Image(z=matrix, coloraxis="coloraxis"),
        row=row, col=col
    )

# Optional: Customize layout
# fig.update_layout(
#     coloraxis=dict(colorscale='Viridis'), # Use same color scale for all plots
#     title_text="Multiple 12x12 Heat Maps"
# )

# Show figure
fig.show()

# %%
