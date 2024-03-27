# %%
import random
from setup import *
from datasets import load_dataset

torch = t


# %%
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 32,
    "seq_len": 128,
    "enc_dtype": "fp32",
    "remove_rare_dir": False,
    "model_name": "gelu-2l",
    "site": "mlp_out",
    "layer": 0,
    "device": "cuda:0",
}

cfg = default_cfg

site_to_size = {
    "mlp_out": 512,
    "post": 2048,
    "resid_pre": 512,
    "resid_mid": 512,
    "resid_post": 512,
}


def post_init_cfg(cfg):
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
    cfg["act_name"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["act_size"] = site_to_size[cfg["site"]]
    cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
    cfg["name"] = f"{cfg['model_name']}_{cfg['layer']}_{cfg['dict_size']}_{cfg['site']}"


post_init_cfg(cfg)


# %%
SEED = cfg["seed"]
GENERATOR = torch.manual_seed(SEED)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(True)


# %%
model = (
    HookedTransformer.from_pretrained(cfg["model_name"])
    .to(DTYPES[cfg["enc_dtype"]])
    .to(cfg["device"])
)

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab


# %%
@torch.no_grad()
def get_acts(tokens, batch_size=1024):
    _, cache = model.run_with_cache(
        tokens, stop_at_layer=cfg["layer"] + 1, names_filter=cfg["act_name"]
    )
    acts = cache[cfg["act_name"]]
    acts = acts.reshape(-1, acts.shape[-1])
    subsample = torch.randperm(acts.shape[0], generator=GENERATOR)[:batch_size]
    subsampled_acts = acts[subsample, :]
    return subsampled_acts, acts


# %%
act1 = get_acts(model.to_tokens("Hello"))[0]


# %%
def shuffle_data(all_tokens):
    print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]


# %%
loading_data_first_time = False
if loading_data_first_time:
    data = load_dataset(
        "NeelNanda/c4-code-tokenized-2b", split="train", cache_dir="/workspace/cache/"
    )
    data.save_to_disk("/workspace/data/c4_code_tokenized_2b.hf")
    data.set_format(type="torch", columns=["tokens"])
    all_tokens = data["tokens"]
    all_tokens.shape

    all_tokens_reshaped = einops.rearrange(
        all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128
    )
    all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
    all_tokens_reshaped = all_tokens_reshaped[
        torch.randperm(all_tokens_reshaped.shape[0])
    ]
    torch.save(all_tokens_reshaped, "/workspace/data/c4_code_2b_tokens_reshaped.pt")
else:
    # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
    print("Loading tokens from disk")
    all_tokens = torch.load("/workspace/data/c4_code_2b_tokens_reshaped.pt")
    all_tokens = shuffle_data(all_tokens)


# %%
class Buffer:
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg):
        self.buffer = torch.zeros(
            (cfg["buffer_size"], cfg["act_size"]),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.cfg["buffer_batches"]
            else:
                num_batches = self.cfg["buffer_batches"] // 2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = all_tokens[
                    self.token_pointer : self.token_pointer
                    + self.cfg["model_batch_size"]
                ]
                _, cache = model.run_with_cache(
                    tokens, stop_at_layer=cfg["layer"] + 1, names_filter=cfg["act_name"]
                )
                acts = cache[cfg["act_name"]].reshape(-1, self.cfg["act_size"])

                # print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
                # if self.token_pointer > all_tokens.shape[0] - self.cfg["model_batch_size"]:
                #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(cfg["device"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out
