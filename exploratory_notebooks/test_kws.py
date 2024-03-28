# %%
%load_ext autoreload
%autoreload 2


# %%
from k_winner_synthesis import KWinnerSynthesis, KWSConfig, cfg as buffer_config

# %%
simple_config = KWSConfig(
    num_features=4_000,
    n_winners=20,
    num_batches=100,
    lr=0.001,
    lr_schedule=[(20, 0.001), (100, 0.005)],
)

# %%
kws = KWinnerSynthesis(config=simple_config, buffer_config=buffer_config)

# %%
kws.train(1000)

# %%
kws.features