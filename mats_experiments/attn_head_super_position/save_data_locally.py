# %%
%load_ext autoreload
%autoreload 2


# %%
import torch
import os

from attribution_buffer import load_attribution_tensor
from generate_data import get_file_name
from tqdm import trange
from pathlib import Path

# %%
SAVE_DIR = "/workspace/data/attribution/"
save_dir_path = Path(SAVE_DIR)

os.makedirs(save_dir_path, exist_ok=True)

# %%
ATTR_TYPE = 'bos'

for batch_i in trange(64 * 4):
    ten = load_attribution_tensor(ATTR_TYPE, batch_i, 1000)
    file_name = get_file_name(ATTR_TYPE, batch_i, 1000)

    torch.save(ten, SAVE_DIR + file_name)


# %%
