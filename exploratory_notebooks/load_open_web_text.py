# %%
%load_ext autoreload
%autoreload 2

# %%
from datasets import load_dataset

# %%
data = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train", cache_dir="/workspace/cache/")
data.set_format(type="torch", columns=["input_ids"])

# %%
data.format

# %%
tokens = data['input_ids']

# %%
data.select(range(10))['input_ids'].device

# %%
data

# %%
