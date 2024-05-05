# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import boto3
import pathlib
import io

from tqdm import trange

# %%


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


# %%
session = boto3.Session(
    # aws_access_key_id='',
    # aws_secret_access_key='',
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
import requests

# %%
url = 'https://mech-interp.s3.us-east-2.amazonaws.com/attribution/attr-type-mix-batch-start-0-batch-size-3.pt'

res = requests.get(url)

tensor = torch.load(io.BytesIO(res.content))

# %%
tensor.shape

# %%
torch.cuda.device_count()

# %%
