import os
import sys
import plotly.express as px
import torch as t
import torch

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv


class AttributionSAE(nn.Module):
    def __init__(
        self, attr_type: str, n_features: int, dtype=torch.bfloat16, act_size=144
    ):
        super().__init__()

        self.attr_type = attr_type
        self.act_size = act_size
        self.n_features = n_features

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
