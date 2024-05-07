import requests
import torch
import io
import einops


from data import select_token_range
from generate_data import generate_data

from attention_attribution import (
    get_bos_ablate_for_head,
)

from transformer_lens import HookedTransformer

from generate_data import get_file_name

from d_types import DTYPES

s3_url_template = "https://mech-interp.s3.us-east-2.amazonaws.com/attribution/{}"
SAVE_DIR = "/workspace/data/attribution/"


def load_attribution_tensor(attr_type: str, batch_i: int, batch_size=1000):
    file_name = get_file_name(attr_type, batch_i, batch_size)
    s3_url = s3_url_template.format(file_name)

    res = requests.get(s3_url)
    return torch.load(io.BytesIO(res.content))


def load_attribution_tensor_locally(attr_type: str, batch_i: int, batch_size=1000):
    return torch.load(SAVE_DIR + get_file_name(attr_type, batch_i, batch_size))


def load_bos_ablate_for_head():
    cfg = {
        "model": "gpt2",
        "device": f"cuda:{0}",
        "enc_dtype": "bf16",
    }

    model = (
        HookedTransformer.from_pretrained(cfg["model"])
        .to(DTYPES[cfg["enc_dtype"]])
        .to(cfg["device"])
    )

    test_tokens_for_bos_ablate = select_token_range(0, 100).to(cfg["device"])

    bos_ablate_for_head = get_bos_ablate_for_head(
        model,
        test_tokens_for_bos_ablate,
        num_samples=100,
        k=3,
        bos_value_compare_ratio=0.1,
        bos_ablate_threshold=0.75,
    )

    return bos_ablate_for_head, model


class AttributionBuffer:
    data_batch_size = None

    def __init__(
        self,
        attr_type: str,
        batch_size: int,
        reshape=True,
        device="cuda:0",
        local=True,
        normalize=False,
        mask=False,
    ):
        self.attr_type = attr_type
        self.batch_size = batch_size
        self.batch_i = -1
        self.reshape = reshape
        self.device = device
        self.local = local
        self.normalize = normalize

        self.attribution_data_index = -1
        self.attribution_data = None

        self.load_next_attribution_data()

        self.bos_ablate_for_head = None

        if mask:
            bos, model = load_bos_ablate_for_head()

            self.bos_ablate_for_head = bos
            self.model = model

    def load_next_attribution_data(self):
        self.attribution_data_index += 1

        print("Fetching data:", self.attribution_data_index)

        if self.local:
            self.attribution_data = load_attribution_tensor_locally(
                self.attr_type, self.attribution_data_index
            )
        else:
            self.attribution_data = load_attribution_tensor(
                self.attr_type, self.attribution_data_index
            )

        if self.data_batch_size is None:
            self.data_batch_size = self.attribution_data.shape[0]
            self._max_batches_per_data = None

    _max_batches_per_data = None

    def mask_data_if_available(self, data):
        if self.bos_ablate_for_head is None:
            return data

        for layer, head in torch.nonzero(
            torch.logical_not(self.bos_ablate_for_head)
        ).tolist():
            data[:, layer, head] = 0

    @property
    def max_batches_per_data(self):
        if self._max_batches_per_data is None:
            assert isinstance(self.data_batch_size, int)
            self._max_batches_per_data = self.data_batch_size // self.batch_size

        return self._max_batches_per_data

    def next(self):
        self.batch_i += 1

        if self.batch_i >= self.max_batches_per_data:
            self.load_next_attribution_data()
            self.batch_i = 0

        if self.attribution_data is None:
            raise ValueError("No attribution data loaded")

        batch_data = self.attribution_data[
            self.batch_i * self.batch_size : (self.batch_i + 1) * self.batch_size
        ].to(self.device)

        if self.bos_ablate_for_head is not None:
            self.mask_data_if_available(batch_data)
            # for layer, head in torch.nonzero(
            #     torch.logical_not(self.bos_ablate_for_head)
            # ).tolist():
            #     batch_data[:, layer, head] = 0

        if self.reshape:
            batch_data = einops.rearrange(batch_data, "N l h -> N (l h)")

            if self.normalize:
                batch_data /= batch_data.norm(dim=-1, keepdim=True)

            return batch_data
        else:
            return batch_data
