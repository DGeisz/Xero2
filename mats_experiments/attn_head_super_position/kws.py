import torch as t
import torch
import einops
import time
import tqdm
import circuitsvis as cv

from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional
from jaxtyping import Float, Int
from attribution_buffer import load_attribution_tensor_locally
from data import select_token_range

device = "cuda:0"


def hyperbola(x, a):
    return a * (1 + (x / a) ** 2) ** 0.5


@dataclass
class KWSConfig:
    attr_type: str
    device = "cuda:0"

    num_features: int
    n_winners: int
    num_batches: int

    lr: float
    lr_schedule: Optional[List[Tuple[int, float]]] = None

    hyp_min = 0.001
    update_mini_batch_size: int = 200

    ## Logging
    log_freq: Optional[Int] = 10

    ## Dead Neuron Update
    # How frequently to update dead neurons (in batches)
    update_dead_neurons_freq: Optional[Int] = None
    # What fraction of the top feature do we want to add to dead neurons?
    dead_neuron_resample_fraction: float = 1

    # Number of batches to wait before resampling dead neurons
    dead_neuron_resample_initial_delay: int = 20

    mask_initial_neurons: bool = False


@dataclass
class KWSSingleBatchOutput:
    data: Float[Tensor, "N D"]
    reconstructed_data: Float[Tensor, "N D"]
    max_data_magnitude: Float[Tensor, "N D"]

    feature_output: Float[Tensor, "N F"]
    max_output: Float[Tensor, "F 1"]

    mask: Int[Tensor, "N F"]
    winner_count: Int[Tensor, "F"]

    winner_features: Optional[Float[Tensor, "M D"]] = None


MIN_PREC = 1e-10


class KWinnerSynthesis:
    width = None
    features: Float[Tensor, "num_features width"]

    def __init__(self, config: KWSConfig, buffer):
        self.config = config
        self.batch_index = 0
        self.batch_size = buffer.batch_size

        self._buffer = buffer
        self._init_weights()

    def random_data_batch(self):
        return self._buffer.next()

    def _init_weights(self):
        self.batch = 0

        if not self.width:
            _, width = self.random_data_batch().shape
            self.width = width

        all_data = []

        for _ in range((self.config.num_features // self.batch_size) + 1):
            all_data.append(self.random_data_batch())

        features = t.cat(all_data, dim=0)[: self.config.num_features, : self.width]
        features = features / features.norm(dim=-1, keepdim=True)

        self.features = features

    def get_winners(self, data, n_winners=None):
        if n_winners is None:
            n_winners = self.config.n_winners

        data = self.mask_data_if_applicable(data)
        data = data / data.norm(dim=-1).unsqueeze(-1)

        raw_output = einops.einsum(data, self.features, "n d, f d -> n f")

        values, indices = t.sort(raw_output, descending=True, dim=-1)

        return (
            indices[:, :n_winners],
            values[:, :n_winners],
        )

    def reconstruct(self, data):
        data = self.mask_data_if_applicable(data)

        raw_output = einops.einsum(data, self.features, "n d, f d -> n f")
        winner_indices = t.argsort(raw_output, descending=True, dim=-1)[
            :, : self.config.n_winners
        ]

        # We want to set all values that weren't winners to 0
        mask = t.zeros_like(raw_output)
        winner_rows = (
            t.arange(winner_indices.size(0)).unsqueeze(1).expand_as(winner_indices)
        )
        mask[winner_rows, winner_indices] = 1

        winner_count = mask.sum(dim=0)

        feature_output = mask * raw_output
        max_output = feature_output.max(dim=-1).values.unsqueeze(1)

        reconstructed_data = einops.einsum(
            feature_output, self.features, "n f, f d -> n d"
        )

        return reconstructed_data

    def mask_data_if_applicable(self, data):
        if self.config.mask_initial_neurons:
            data = data.clone()
            data[:, :96] = 0
            data = data / data.norm(dim=-1, keepdim=True)

        return data

    def get_winner_indices_and_values(self, data):
        data = self.mask_data_if_applicable(data)

        raw_output = einops.einsum(data, self.features, "n d, f d -> n f")

        values, indices = t.sort(raw_output, descending=True, dim=-1)
        # print("eta nu", values.shape, indices.shape)
        # [
        #     :, : self.config.n_winners
        # ]

        # winner_indices = t.argsort(raw_output, descending=True, dim=-1)[
        #     :, : self.config.n_winners
        # ]

        return values[:, : self.config.n_winners], indices[:, : self.config.n_winners]

    def get_features_for_winners(self, winner_indices):
        return self.features[winner_indices]

    def run_and_reconstruct_single_batch(
        self, include_winner_features=False
    ) -> KWSSingleBatchOutput:
        data = self.random_data_batch()

        data = self.mask_data_if_applicable(data)

        raw_output = einops.einsum(data, self.features, "n d, f d -> n f")
        winner_indices = t.argsort(raw_output, descending=True, dim=-1)[
            :, : self.config.n_winners
        ]

        winner_features = None

        if include_winner_features:
            winner_features = self.features[winner_indices]

        # We want to set all values that weren't winners to 0
        mask = t.zeros_like(raw_output)
        winner_rows = (
            t.arange(winner_indices.size(0)).unsqueeze(1).expand_as(winner_indices)
        )
        mask[winner_rows, winner_indices] = 1

        winner_count = mask.sum(dim=0)

        feature_output = mask * raw_output
        max_output = feature_output.max(dim=-1).values.unsqueeze(1)

        reconstructed_data = einops.einsum(
            feature_output, self.features, "n f, f d -> n d"
        )

        max_data_magnitude = t.maximum(reconstructed_data.abs(), data.abs())
        max_data_magnitude = t.maximum(max_data_magnitude, t.tensor(MIN_PREC))

        return KWSSingleBatchOutput(
            data=data,
            reconstructed_data=reconstructed_data,
            max_data_magnitude=max_data_magnitude,
            feature_output=feature_output,
            max_output=max_output,
            mask=mask,
            winner_count=winner_count,
            winner_features=winner_features,
        )

    def get_winner_count_from_n_batches(self, batches: Int):
        winner_count = t.zeros(self.config.num_features, device=device)

        for _ in range(batches):
            output = self.run_and_reconstruct_single_batch()
            winner_count += output.winner_count

        return winner_count

    def _get_lr(self, batch: Int):
        if self.config.lr_schedule:
            for step, lr in self.config.lr_schedule:
                if batch < step:
                    return lr

        return self.config.lr

    def re_init_weights_and_train(self, num_batches: Optional[Int] = None):
        self._init_weights()
        self.train(num_batches)

    def train(self, num_batches: Optional[Int] = None):
        if num_batches is None:
            num_batches = self.config.num_batches

        start_batch_index = self.batch_index

        start_time = time.time()
        cumulative_error = 0
        winner_count = t.zeros(self.config.num_features, device=device)

        dead_neuron_winner_count = t.zeros(self.config.num_features, device=device)

        for batch in range(start_batch_index, start_batch_index + num_batches):
            self.batch_index += 1

            run_output = self.run_and_reconstruct_single_batch()
            o = run_output

            lr = self._get_lr(batch)

            error = run_output.data - run_output.reconstructed_data

            if self.config.log_freq is not None:
                winner_count += run_output.winner_count
                cumulative_error += error.pow(2).sum().item()

            if self.config.update_dead_neurons_freq is not None:
                dead_neuron_winner_count += run_output.winner_count

            # Update the feature matrix in minibatches to not use too much GPU memory
            for indices in t.split(
                t.arange(self.batch_size), self.config.update_mini_batch_size
            ):
                self.features += (
                    hyperbola(self.features, self.config.hyp_min).unsqueeze(0)
                    * (o.feature_output / o.max_output)[indices, :].unsqueeze(-1)
                    * (error / o.max_data_magnitude)[indices, :].unsqueeze(-2)
                ).sum(dim=0) * lr

            ## Log training details
            if (
                self.config.log_freq
                and batch > start_batch_index
                and batch % self.config.log_freq == 0
            ):
                print(
                    f"Time: {time.time() - start_time:.2f}, Batch: {batch}, "
                    + f"Error: {cumulative_error / (self.config.log_freq * self.batch_size)}, "
                    + f"Num Winners: {(winner_count > 0).sum().item()}, "
                    + f"Avg Winners: {winner_count.mean().item()}, "
                    + f"Above Average {((winner_count > winner_count.mean()).sum().item())}"
                )

                winner_count = t.zeros(self.config.num_features, device=device)
                cumulative_error = 0

            ## Update dead neurons
            if (
                self.config.update_dead_neurons_freq is not None
                and (batch % self.config.update_dead_neurons_freq == 0)
                and batch > self.config.dead_neuron_resample_initial_delay
            ):
                dead_indices = t.where(dead_neuron_winner_count == 0)[0]
                most_popular_feature = self.features[
                    int(dead_neuron_winner_count.argmax().item())
                ]

                dead_neuron_mask = t.zeros(self.config.num_features, device=device)
                dead_neuron_mask[dead_indices] = 1

                update = einops.einsum(
                    dead_neuron_mask, most_popular_feature, "f, d -> f d"
                )
                self.features += update * self.config.dead_neuron_resample_fraction

                print(f"Updated {dead_indices.shape[0]} dead neurons")

                dead_neuron_winner_count = t.zeros(
                    self.config.num_features, device=device
                )

    sequence_attribution = None
    seq_attr_N = None

    def get_sequence_attribution(self, N=None):
        attr_data = load_attribution_tensor_locally(self.config.attr_type, 0, 1000)
        seq_len = 119

        attr_data = einops.rearrange(
            attr_data, "(batch seq) l h -> batch seq l h", seq=seq_len
        )

        process_batch_size = 32

        sequence_attribution = [[] for _ in range(self.config.num_features)]

        if N is None:
            N = attr_data.shape[0]

        total = N // process_batch_size

        for i, indices in tqdm.tqdm(
            enumerate(torch.split(torch.arange(N), process_batch_size)),
            total=total,
        ):
            data = attr_data[indices]

            if len(indices) != process_batch_size:
                break

            reshaped_data = einops.rearrange(data, "batch seq l h -> (batch seq) (l h)")
            reshaped_data /= reshaped_data.norm(dim=-1, keepdim=True)

            winner_values, winner_indices = self.get_winner_indices_and_values(
                reshaped_data.to(self.config.device)
            )

            winner_indices = einops.rearrange(
                winner_indices, "(batch seq) n -> batch seq n", seq=seq_len
            )
            winner_values = einops.rearrange(
                winner_values, "(batch seq) n -> batch seq n", seq=seq_len
            )

            for batch in range(process_batch_size):
                for seq_i in range(seq_len):
                    for winner_i, value in zip(
                        winner_indices[batch, seq_i].tolist(),
                        winner_values[batch, seq_i].tolist(),
                    ):
                        sequence_attribution[winner_i].append(
                            ((i * process_batch_size) + batch, seq_i + 9, value)
                        )

        self.sequence_attribution = sequence_attribution
        self.seq_attr_N = N

        return sequence_attribution

    _tokens = None

    def show_fucking_anything(self, feature_i):
        tokens = select_token_range(0, self.seq_attr_N)
