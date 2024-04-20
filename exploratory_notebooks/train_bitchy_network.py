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

buffer = Buffer(cfg)

# %%
bitchy_network = BitchyNetwork(cfg, 20_000, 20, buffer)


# %%
def replacement_hook(mlp_post, hook, encoder):
    n, s, d = mlp_post.shape

    mlp_post = einops.rearrange(mlp_post, "n s d -> (n s) d")

    mlp_post_reconstr = encoder(mlp_post)[1]

    mlp_post_reconstr = einops.rearrange(mlp_post_reconstr, "(n s) d -> n s d", n=n, s=s)

    print(mlp_post_reconstr.shape, mlp_post.shape)


    return mlp_post_reconstr
    # return mlp_post

def nothing_hook(mlp_post, hook):
    return mlp_post


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


@torch.no_grad()
def get_recons_loss_2(num_batches=5, local_encoder=None):
    num_batches = 5
    local_encoder = bitchy_network

    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[: cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")

        recons_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[
                (cfg["act_name"], partial(replacement_hook, encoder=local_encoder))
            ],
        )

        zero_abl_loss = model.run_with_hooks(
            tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], zero_ablate_hook)]
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss


# %%
cfg['enc_dtype']

# %%
cfg['lr'] = 1e-4



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
    if (i) % 500 == 0:
        x = get_recons_loss_2(local_encoder=bitchy_network)
        print("Reconstruction:", x)
    #     recons_scores.append(x[0])
    #     freqs = get_freqs(5, local_encoder=bitchy_network)
    #     act_freq_scores_list.append(freqs)
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
get_recons_loss(local_encoder=bitchy_network)


# %%
@torch.no_grad()
def get_freqs(num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.W.shape[0], dtype=torch.float32).to(
        cfg["device"]
    )
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[: cfg["model_batch_size"]]]

        _, cache = model.run_with_cache(
            tokens, stop_at_layer=cfg["layer"] + 1, names_filter=cfg["act_name"]
        )
        acts = cache[cfg["act_name"]]
        acts = acts.reshape(-1, cfg["act_size"])

        hidden = local_encoder(acts)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total += hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores == 0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores

freqs = get_freqs(5, local_encoder=bitchy_network)


# %%
(freqs == 0).sum()
# %%
freqs.max()


# %%
