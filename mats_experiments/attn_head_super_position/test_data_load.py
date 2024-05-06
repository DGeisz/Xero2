# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import boto3
import pathlib
import io
import requests


from tqdm import trange
import plotly.express as px

from generate_data import file_template
from transformer_lens import utils

# %%
def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()


# %%
a = torch.rand(117 * 10, 144)

# %%

b = []

for _ in trange(1000):
    b.append(
        torch.rand(117 * 10, 12, 12)
    )


# %%
bb = torch.concat(b, dim=0)

# %%
bb.shape


# %%
bb.dtype

# %%
(bb.numel() * 4) / 1e9


# %%
bb.shape

# %%
torch.save(bb, 'test.pt')

# bws_access_key_id='',
# bws_secret_access_key='',

# %%
session = boto3.Session(
    region_name='us-east-2'
)
s3 = session.client('s3')

# %%
tensor_content = s3.get_object(Bucket="mech-interp", Key="attribution/attr-type-bos-batch-start-0-batch-size-3.pt")['Body'].read()

# %%
tensor_stream = io.BytesIO(tensor_content)

tensor = torch.load(tensor_stream)

# %%
tensor


# %%
s3.upload_file('test.pt', 'mech-interp', 'attr/test.pt')



# %%
pathlib.Path.unlink('test.pt')

# %%

file_name = file_template.format('zero', 1, 125)

# %%
url = f'https://mech-interp.s3.us-east-2.amazonaws.com/attribution/{file_name}'

res = requests.get(url)

tensor = torch.load(io.BytesIO(res.content))

# %%
tensor.shape

# %%
imshow(tensor[1000].float())



# %%
tensor.shape

# %%
torch.cuda.device_count()

# %%
