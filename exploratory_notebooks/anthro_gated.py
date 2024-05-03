# %%
%load_ext autoreload
%autoreload 2

# %%
from generate_activations import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
import wandb

# %%
buffer = Buffer(cfg)

# %%
class AnthroGatedSAE(nn.Module):
    def __init__(self, cfg, n_features=5000, l1_coeff=None):
        super().__init__()

        d_hidden = n_features

        if l1_coeff is not None:
            self.l1_coeff = l1_coeff
        else:
            self.l1_coeff = cfg["l1_coeff"]

        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])

        self.W_gate = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg["act_size"], d_hidden, dtype=dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_hidden, cfg["act_size"], dtype=dtype)
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.b_enc_gate = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec_gate = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.r_mag = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_mag = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))

        self.b_gate = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden

        self.to(cfg["device"])


    def gated_sae(self, x):
        preactivations_hidden = einops.einsum(x, self.W_gate, "... input_dim, input_dim hidden_dim -> ... hidden_dim")

        pre_mag_hidden = preactivations_hidden * torch.exp(self.r_mag) + self.b_mag
        post_mag_hidden = torch.relu(pre_mag_hidden)

        pre_gate_hidden = preactivations_hidden + self.b_gate
        post_gate_hidden = (torch.sign(pre_gate_hidden) + 1) / 2

        postactivations_hidden = post_mag_hidden * post_gate_hidden


        reconstruction =  einops.einsum(postactivations_hidden, self.W_dec, "... hidden_dim, hidden_dim output_dim -> ... output_dim") + self.b_dec

        return reconstruction, pre_gate_hidden
        



    def forward(self, x):
        reconstruction, pre_gate_hidden = self.gated_sae(x)

        # Reconstruction Loss
        l2_loss = (reconstruction - x).pow(2).sum(-1).mean(0)

        gated_sae_loss = l2_loss.clone()

        # L1 loss
        gate_magnitude = F.relu(pre_gate_hidden)

        # l1_loss = self.l1_coeff * gate_magnitude.sum()
        l1_loss = self.l1_coeff * (gate_magnitude * self.W_dec.norm(dim=-1).unsqueeze(0)).sum()
        l0 = (gate_magnitude > 0).sum() / x.shape[0]

        gated_sae_loss += l1_loss

        # Auxiliary loss
        gate_reconstruction = einops.einsum(gate_magnitude, self.W_dec.detach(), "... hidden_dim, hidden_dim output_dim -> ... output_dim") + self.b_dec.detach()
        auxiliary_loss = F.mse_loss(gate_reconstruction, x, reduction='mean')

        gated_sae_loss += auxiliary_loss

        return gated_sae_loss, reconstruction, gate_magnitude, l2_loss, l1_loss, auxiliary_loss, l0


    # @torch.no_grad()
    # def make_decoder_weights_and_grad_unit_norm(self):
    #     W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
    #     W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
    #         -1, keepdim=True
    #     ) * W_dec_normed
    #     self.W_dec.grad -= W_dec_grad_proj
    #     # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.



# %%
encoder = AnthroGatedSAE(cfg, 5000, l1_coeff=1e-4)
model_dtype = DTYPES[cfg["enc_dtype"]]



# %%
num_batches = cfg["num_tokens"] // cfg["batch_size"]

encoder_optim = torch.optim.Adam(
    encoder.parameters(), 
    lr=1e-4,
    betas=(cfg["beta1"], cfg["beta2"])
)

# wandb.init(project="Xero")
for i in range(num_batches):
    i = i % all_tokens.shape[0]

    acts = buffer.next()

    loss, reconstruction, gate_magnitude, l2_loss, l1_loss, auxiliary_loss, l0 = encoder(acts)
    loss.backward()

    loss_dict = {
        "loss": loss.item(),
        "l2_loss": l2_loss.item(),
        "l1_loss": l1_loss.item(),
        "l0_avg": l0.item()
    }

    # encoder.make_decoder_weights_and_grad_unit_norm()
    encoder_optim.step()
    encoder_optim.zero_grad()

    if (i) % 100 == 0:
        # wandb.log(loss_dict)
        print(f"Loss: {loss.item():.2f}, L2: {l2_loss.item():.2f}, L1: {l1_loss.item():.2f}, Aux: {auxiliary_loss.item():.2f}, L0: {l0.item():.2f}")


    del loss, l2_loss, l1_loss, auxiliary_loss, l0, reconstruction, gate_magnitude

    if (i) % 1000 == 0:
        x = get_recons_loss(local_encoder=encoder)
        print("Reconstruction:", x)
        freqs = get_freqs(5, local_encoder=encoder)
        # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
        # wandb.log(
        #     {
        #         "recons_score": x[0],
        #         "dead": (freqs == 0).float().mean().item(),
        #         "below_1e-6": (freqs < 1e-6).float().mean().item(),
        #         "below_1e-5": (freqs < 1e-5).float().mean().item(),
        #     }
        # )
# %%
