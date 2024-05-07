import torch

import torch.nn as nn
import torch.nn.functional as F


class AttributionSAE(nn.Module):
    def __init__(
        self,
        attr_type: str,
        n_features: int,
        l1_coeff: float,
        dtype=torch.float32,
        act_size=144,
        device="cuda:0",
    ):
        super().__init__()

        self.attr_type = attr_type
        self.act_size = act_size
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.act_size, self.n_features, dtype=dtype)
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.act_size, self.n_features, dtype=dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.n_features, self.act_size, dtype=dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(self.n_features, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(self.act_size, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to(device)

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)

        x_reconstruct = acts @ self.W_dec + self.b_dec

        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        l0 = (acts > 0).sum() / x.shape[0]

        loss = l2_loss + l1_loss

        return loss, x_reconstruct, acts, l2_loss, l1_loss, l0

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
