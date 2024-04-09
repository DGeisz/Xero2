# %%
%load_ext autoreload
%autoreload 2

# %%
import tqdm
from generate_activations import *
import wandb
from bitchy_network import BitchyNetwork


# %%
cfg['d_mlp'] = model.cfg.d_mlp

bitchy_network = BitchyNetwork(cfg, 10_000, 10)
buffer = Buffer(cfg)


# %%
bitchy_network.W.device

# %%



# wandb.init(project="Xero")
num_batches = cfg["num_tokens"] // cfg["batch_size"]
# model_num_batches = cfg["model_batch_size"] * num_batches
encoder_optim = torch.optim.Adam(
    bitchy_network.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
)
recons_scores = []
act_freq_scores_list = []
for i in tqdm.trange(num_batches):
    i = i % all_tokens.shape[0]
    acts = buffer.next()
    loss, x_rec, acts = bitchy_network(acts.to(torch.float32))
    loss.backward()
    # bitchy_network.make_decoder_weights_and_grad_unit_norm()
    encoder_optim.step()
    encoder_optim.zero_grad()
    loss_dict = {
        "loss": loss.item(),
        # "l2_loss": l2_loss.item(),
        # "l1_loss": l1_loss.item(),
    }
    del loss, x_rec, acts
    if (i) % 100 == 0:
        # wandb.log(loss_dict)
        print(loss_dict)
    if (i) % 1000 == 0:
        x = get_recons_loss(local_encoder=bitchy_network)
        print("Reconstruction:", x)
        recons_scores.append(x[0])
        freqs = get_freqs(5, local_encoder=bitchy_network)
        act_freq_scores_list.append(freqs)
        # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
        # wandb.log(
        #     {
        #         "recons_score": x[0],
        #         "dead": (freqs == 0).float().mean().item(),
        #         "below_1e-6": (freqs < 1e-6).float().mean().item(),
        #         "below_1e-5": (freqs < 1e-5).float().mean().item(),
        #     }
        # )
    # if (i + 1) % 30000 == 0:
    #     bitchy_network.save()
    #     wandb.log({"reset_neurons": 0.0})
    #     freqs = get_freqs(50, local_encoder=bitchy_network)
    #     to_be_reset = freqs < 10 ** (-5.5)
    #     print("Resetting neurons!", to_be_reset.sum())
    #     re_init(to_be_reset, bitchy_network)

# %%
bitchy_network.W.dtype

# %%
data = buffer.next()

# %%

