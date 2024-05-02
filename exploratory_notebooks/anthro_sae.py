# %%
%load_ext autoreload
%autoreload 2

# %%
from generate_activations import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import einops

# %%
buffer = Buffer(cfg)

# %%
class AnthroSAE(nn.Module):
    def __init__(self, cfg, d_hidden=5000, l1_coeff=3e-4):
        super().__init__()

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        act_size = cfg['act_size']

        # d_hidden = cfg["dict_size"]
        # l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]

        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(act_size, d_hidden, dtype=dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_hidden, act_size, dtype=dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(act_size, dtype=dtype))

        # self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)


        self.to(cfg["device"])

    def forward(self, x):
        acts = F.relu(x @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec

        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        # l1_loss = self.l1_coeff * (acts.float().abs().sum())
        l1_loss = self.l1_coeff * (acts.float().abs() * self.W_dec.norm(dim=-1).unsqueeze(0)).sum()
        l0 = (acts > 0).sum() / x.shape[0]
        # l1_loss = self.l1_coeff * einops.einsum(acts, , "batch h_dim")

        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss, l0

    # @torch.no_grad()
    # def make_decoder_weights_and_grad_unit_norm(self):
    #     W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
    #     W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
    #         -1, keepdim=True
    #     ) * W_dec_normed
    #     self.W_dec.grad -= W_dec_grad_proj
    #     # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
    #     self.W_dec.data = W_dec_normed

    # def get_version(self):
    #     version_list = [
    #         int(file.name.split(".")[0])
    #         for file in list(SAVE_DIR.iterdir())
    #         if "pt" in str(file)
    #     ]
    #     if len(version_list):
    #         return 1 + max(version_list)
    #     else:
    #         return 0

    # def save(self):
    #     version = self.get_version()
    #     torch.save(self.state_dict(), SAVE_DIR / (str(version) + ".pt"))
    #     with open(SAVE_DIR / (str(version) + "_cfg.json"), "w") as f:
    #         json.dump(cfg, f)
    #     print("Saved as version", version)

    # @classmethod
    # def load(cls, version):
    #     cfg = json.load(open(SAVE_DIR / (str(version) + "_cfg.json"), "r"))
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(torch.load(SAVE_DIR / (str(version) + ".pt")))
    #     return self

    # @classmethod
    # def load_from_hf(cls, version):
    #     """
    #     Loads the saved autoencoder from HuggingFace.

    #     Version is expected to be an int, or "run1" or "run2"

    #     version 25 is the final checkpoint of the first autoencoder run,
    #     version 47 is the final checkpoint of the second autoencoder run.
    #     """
    #     if version == "run1":
    #         version = 25
    #     elif version == "run2":
    #         version = 47

    #     cfg = utils.download_file_from_hf(
    #         "NeelNanda/sparse_autoencoder", f"{version}_cfg.json"
    #     )
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(
    #         utils.download_file_from_hf(
    #             "NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True
    #         )
    #     )
    #     return self

# %%
encoder = AnthroSAE(cfg, 5000, l1_coeff=6e-4)
model_dtype = DTYPES[cfg["enc_dtype"]]

# %%
num_batches = cfg["num_tokens"] // cfg["batch_size"]

# model_num_batches = cfg["model_batch_size"] * num_batches
encoder_optim = torch.optim.Adam(
    encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
)
recons_scores = []
act_freq_scores_list = []
for i in range(num_batches):
    i = i % all_tokens.shape[0]

    # acts = buffer.next().to(model_dtype)
    acts = buffer.next()
    # loss = encoder(acts)

    # loss, l2_loss, l1_loss, auxiliary_loss, l0 = encoder(acts)

    loss, x_reconstruct, acts, l2_loss, l1_loss, l0 = encoder(acts)
    loss.backward()
    # encoder.make_decoder_weights_and_grad_unit_norm()
    encoder_optim.step()
    encoder_optim.zero_grad()

    if (i) % 100 == 0:
        #Print loss values to two decimal places
        print(f"Loss: {loss.item():.2f}, L2: {l2_loss.item():.2f}, L1: {l1_loss.item():.2f}, L0: {l0.item():.2f}")

        # print(loss.item())
        # wandb.log(loss_dict)
        # print(loss_dict)

    del loss, l2_loss, l1_loss, x_reconstruct, acts
    # if (i) % 1000 == 0:



# %%
