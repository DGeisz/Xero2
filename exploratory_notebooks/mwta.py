# %%
from setup import *
from generate_activations import Buffer, model, cfg, default_cfg, post_init_cfg
import time

# %%
buffer_config = default_cfg
buffer_config["batch_size"] = 4096

post_init_cfg(buffer_config)

batch_size = buffer_config["batch_size"]


# %%
d_model = model.cfg.d_model
features = 4_000
prec = 1e-10

n_winners = 20

epoch_num_batches = 1000
num_epochs = 1

lr = 0.001
hyp_min = 0.001

# %%
t.set_grad_enabled(False)


# %%
def hyperbola(x, a):
    return a * (1 + (x / a) ** 2) ** 0.5


# %%
buffer = Buffer(buffer_config)

# %%
buffer.next().shape


# %%
all_data = []

for i in range((features // batch_size) + 1):
    all_data.append(buffer.next())

w = t.cat(all_data, dim=0)[:features, :d_model]


w = w / w.norm(dim=-1, keepdim=True)


print("w.shape:", w.shape)


# %%
i = 0
single = False
iter_spec = 10

train_chump = 40

update_dead_neurons_freq = 40
dead_update = 1
wait_dead_update = 3000


start = time.time()

for ep in range(num_epochs):
    error = 0

    for batch in range(epoch_num_batches):
        data = buffer.next()

        winner_count = t.zeros(features, device=cfg["device"])
        serious_winner_count = t.zeros(features, device=cfg["device"])

        if not single:
            data_sign = t.sign(data)

            # lr = 0.005
            if batch < train_chump:
                lr = 0.002
            else:
                lr = 0.01
            # elif batch < 80:
            #     lr = 0.1
            # else:
            #     lr = 0.001

            N, _ = data.shape

            p = einops.einsum(data, w, "n d, f d -> n f")
            winners = t.argsort(input=p, descending=True, dim=-1)[:, :n_winners]

            mask = t.zeros_like(p)
            rows = t.arange(winners.size(0)).unsqueeze(1).expand_as(winners)
            mask[rows, winners] = 1

            winner_count += mask.sum(dim=0)
            serious_winner_count += mask.sum(dim=0)

            o = mask * p
            mo = o.max(dim=-1).values.unsqueeze(1)

            r = einops.einsum(o, w, "n f, f d -> n d")
            e = data - r

            error += e.abs().sum().item()

            mod_r = t.maximum(r.abs(), data.abs())
            mod_r = t.maximum(mod_r, t.tensor(prec))

            for indices in t.split(t.arange(N), 200):
                w += (
                    hyperbola(w, hyp_min).unsqueeze(0)
                    * (o / mo)[indices, :].unsqueeze(-1)
                    * (e / mod_r)[indices, :].unsqueeze(-2)
                    # * data_sign[indices, :].unsqueeze(-2)
                ).sum(dim=0) * lr

            if (batch % iter_spec) == 0:
                print(
                    f"Time: {time.time() - start:.2f}, Batch: {batch}, Error: {error / (iter_spec * N)}, Num Winners: {(winner_count > 0).sum().item()} Avg Winners: {winner_count.mean().item()} Above Average {((winner_count > winner_count.mean()).sum().item())}"
                )

                winner_count = t.zeros(features, device=cfg["device"])
                error = 0

            if (batch % update_dead_neurons_freq) == 0 and batch > wait_dead_update:
                dead_i = t.where(serious_winner_count == 0)[0]
                big_winner = w[winner_count.argmax().item()]

                dead_spots = t.zeros(features).to(device)
                dead_spots[dead_i] = 1

                update = einops.einsum(dead_spots, big_winner, "f, d -> f d")
                w += update * dead_update

                print(f"Updated {dead_i.shape[0]} dead neurons")

                serious_winner_count = t.zeros(features, device=cfg["device"])
        else:
            N, _ = data.shape
            error = 0

            for n in range(N):
                i += 1
                v = data[n]
                p = w @ v
                winners = t.argsort(p, descending=True, axis=0)[:n_winners]
                mask = t.zeros_like(p)
                mask[winners] = 1
                o = mask * p
                mo = o.max()
                r = w.T @ o
                e = v - r

                error += e.abs().sum().item()

                mod_r = t.maximum(r.abs(), v.abs())
                mod_r = t.maximum(mod_r, t.tensor(prec))
                w += (
                    hyperbola(w, hyp_min)
                    * (o / mo).unsqueeze(1)
                    * (e / mod_r).T
                    # * t.sign(v).T
                )

                if (i % 1000) == 0:
                    print(f"Epoch: {ep}, Batch: {batch}, Example: {n}, Iteration: {i}")

            print("Error", error / 4096)
            error = 0
    break


# %%
print(
    "winner count",
    winner_count.mean().item(),
    winner_count.std().item(),
    (winner_count > 0).sum().item(),
    (winner_count == 1).sum().item(),
)


# %%
winner_count.max().item()


# %%
def generate_update(data, w, n_winners=n_winners, hyp_min=hyp_min, prec=prec):
    data_sign = t.sign(data)

    N, _ = data.shape

    p = einops.einsum(data, w, "n d, f d -> n f")
    winners = t.argsort(p, descending=True, axis=-1)[:, :n_winners]

    mask = t.zeros_like(p)
    rows = t.arange(winners.size(0)).unsqueeze(1).expand_as(winners)
    mask[rows, winners] = 1

    o = mask * p
    mo = o.max(dim=-1).values.unsqueeze(1)

    r = einops.einsum(o, w, "n f, f d -> n d")
    e = data - r

    # error += e.abs().sum().item()

    mod_r = t.maximum(r.abs(), data.abs())
    mod_r = t.maximum(mod_r, t.tensor(prec))

    return (
        (
            # t.sign(w) *
            hyperbola(w, hyp_min).unsqueeze(0)
            * (o / mo)[:100, :].unsqueeze(-1)
            * (e / mod_r)[:100, :].unsqueeze(-2)
            # * data_sign[:100, :].unsqueeze(-2)
        ),
        o,
        e,
        r,
        winners,
        mod_r,
    )

    # for indices in t.split(t.arange(N), 200):
    #     yield (
    #         hyperbola(w, hyp_min).unsqueeze(0)
    #         * (o / mo)[indices, :].unsqueeze(-1)
    #         * (e / mod_r)[indices, :].unsqueeze(-2)
    #         * data_sign[indices, :].unsqueeze(-2)
    #     ),

    #     # w += (
    #     #     hyperbola(w, hyp_min).unsqueeze(0)
    #     #     * (o / mo)[indices, :].unsqueeze(-1)
    #     #     * (e / mod_r)[indices, :].unsqueeze(-2)
    #     #     * data_sign[indices, :].unsqueeze(-2)
    #     # ).mean(dim=0) * lr


# %%
winner_count.std()

# %%
dead_i = t.where(winner_count == 0)[0]

# %%
big_winner = w[winner_count.argmax().item()]

# %%
dead_spots = t.zeros(features).to(device)
dead_spots[dead_i] = 1

# %%
update = einops.einsum(dead_spots, big_winner, "f, d -> f d")

# %%
update[:5]


# %%
winner_count.argmax()

# %%
big_winner = w[winner_count.argmax().item()]

# %%
ordered = t.argsort(winner_count, descending=True)

# %%
li = ordered[-2]

big_loser = w[li].clone()

print(
    "above 0: ",
    (big_loser.abs() > 0.01).sum().item(),
    "Winner Count",
    winner_count[li].item(),
)

# %%
big_loser[big_loser.abs() < 0.01] = 0
big_loser


# %%
(big_winner.abs() > 0.01).sum()


# %%
(winner_count == 0).sum()
# %%
big_winner

# %%
(big_winner.abs() > 0.01).sum()

# %%
bw = big_winner.clone()

# %%
bw[bw.abs() < 0.01] = 0
bw
# %%


# %%
(bw < 0).sum()


# %%
(w > 0.01).sum(dim=1).float().min()


# %%

(w[1896] == 0).sum()


# %%
data = buffer.next()

# %%
u, o, e, r, winners, mod_r = generate_update(data, w, n_winners)

# %%


# %%
win = winners[0].item()
win

# %%
win


# %%
mod_r


# %%
def format_list(l):
    return [f"{x:.4f}" for x in l]


a = 10
# print("o", o[0][win].item())
print()
# print("w", format_list(w[win][:a].tolist()))
print("v", format_list(data[0][:a].tolist()))
print("r", format_list(r[0][:a].tolist()))
print("e", format_list(e[0][:a].tolist()))
# print("u", format_list(u[0][win][:a].tolist()))
# print("mod_r", mod_r)


# %%


# %%
u[0][win][:10]

# %%


# %%
data[0][:10]

# %%
r[0][:10]

# %%
w[win][:10]


# %%
winners[0]


# %%
data = buffer.next()

# %%
v = data[0]


# %%
t.sign(v)
# data.shape


# %%
w.shape

# %%
w

# %%
hyperbola(w, hyp_min)


# %%
p = einops.einsum(data, w, "n d, f d -> n f")

# %%
winners = t.argsort(p, descending=True, axis=-1)[:, :n_winners]

# %%
winners

# %%
winners.shape

# %%
mask = t.zeros_like(p)
rows = t.arange(winners.size(0)).unsqueeze(1).expand_as(winners)

# Use advanced indexing to set the corresponding indices in mask to 1
mask[rows, winners] = 1

# %%
o = mask * p

# %%
mo = o.max(dim=-1).values

# %%
r = einops.einsum(o, w, "n f, f d -> n d")
e = data - r


# %%
hyperbola(w, hyp_min) * (o / mo).unsqueeze(1) * (e / mod_r).T

# %%
mod_r = t.maximum(r.abs(), data.abs())
mod_r = t.maximum(mod_r, t.tensor(prec))


# %%
o.shape

# %%
o.max(dim=-1).values.shape

# %%
mo.shape

# %%
o.shape, mo.shape

# %%
mod_r.shape

# %%
e.shape

# %%
o.shape

# %%
w.shape

# %%
(w.unsqueeze(0) * o[:100, :].unsqueeze(-1) * e[:100, :].unsqueeze(-2)).mean(dim=0).shape

# %%
w.shape, e.shape, o.shape


# %%
o[:100, :].shape, e[:, :100].shape


# %%

for indices in t.split(t.arange(full_OV_circuit.shape[0]), batch_size):

    AB_slice = full_OV_circuit[indices].AB
    total += (t.argmax(AB_slice, dim=1) == indices).float().sum().item()


# %%
o / mo.unsqueeze(1)

# %%
o

# %%
mo

# %%

aw = w.clone()
aw = aw / aw.norm(dim=-1, keepdim=True)

# %%
aw.shape

# %%
a_count = winner_count.clone()

# %%
big_winner = aw[winner_count.argmax().item()]

# %%
dots = einops.einsum(aw, big_winner, "f d, d -> f")


# %%
dots.sort(descending=True)[0].tolist()


# %%
# Sort in descending order
aw[dots.sort(descending=True)[1][:20]][:10, :8]

# %%


# %%
a_count.argmax()

# %%
(big_winner > 0.05).sum()


# %%
big_winner.max()

# %%
(aw.abs() > 0.05).sum(dim=-1).float().mean()


# %%
winner_count.sort()[0][-20:]

# %%
winner_count.sort().values[-500:]

# %%
winner_count.sort().indices[-10:]


# %%
aw[407] @ aw[469]

# %%
print(aw[407] - aw[469])

a = aw[407]
b = aw[469]


# %%
v = 0.05

print(
    (a.abs() > v).sum().item(),
    (b.abs() > v).sum().item(),
    ((a - b).abs() > v).sum().item(),
)

# %%
512**0.5

# %%
v = 0.05
vec = avg

print("Above:", (vec > v).sum().item(), "Below:", (vec < -v).sum().item())

# %%
data = buffer.next()

# %%
data.shape

# %%
avg = data.mean(dim=0)

# %%
avg


# %%
avg.std()

# %%
