import requests
import torch
import io
import einops

from generate_data import get_file_name

s3_url_template = "https://mech-interp.s3.us-east-2.amazonaws.com/attribution/{}"


def load_attribution_tensor(attr_type: str, batch_i: int, batch_size=125):
    file_name = get_file_name(attr_type, batch_i, batch_size)
    s3_url = s3_url_template.format(file_name)

    res = requests.get(s3_url)
    return torch.load(io.BytesIO(res.content))


class AttributionBuffer:
    data_batch_size = None

    def __init__(self, attr_type: str, batch_size: int, reshape=True):
        self.attr_type = attr_type
        self.batch_size = batch_size
        self.batch_i = -1
        self.reshape = reshape

        self.attribution_data_index = -1
        self.attribution_data = None

        self.load_next_attribution_data()

    def load_next_attribution_data(self):
        self.attribution_data_index += 1

        self.attribution_data = load_attribution_tensor(
            self.attr_type, self.attribution_data_index
        )

        if self.data_batch_size is None:
            self.data_batch_size = self.attribution_data.shape[0]
            self._max_batches_per_data = None

    _max_batches_per_data = None

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
        ]

        if self.reshape:
            return einops.rearrange(batch_data, "N l h -> N (l h)")
        else:
            return batch_data
