import time


from data import select_token_range 
from generate_data import generate_data

import boto3
import sys

from transformer_lens import HookedTransformer

from attention_attribution import (
    get_bos_ablate_for_head,
)

from d_types import DTYPES

rank = int(sys.argv[1])
batch_start = int(sys.argv[2])

cfg = {
    "model": "gpt2",
    "device": f"cuda:{rank}",
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


session = boto3.Session(
    aws_access_key_id='AKIA2JACKDS2BWWPSBK5',
    aws_secret_access_key='aGZcOM2u7lotE0cdq5i/zCCwvdQwovxXufCqrFtK',
    region_name="us-east-2",
)
s3 = session.client("s3")

big_start = time.time()

generate_data(
    s3_client=s3,
    bos_ablate_for_head=bos_ablate_for_head,
    num_threads=8,
    thread_id=rank,
    seq_batch_size=125,
    # start_batch=8,
    start_batch=batch_start,
    num_batches=8,
    device=f"cuda:{rank}",
    chunk_size=10
)

print()
print("TOTAL TIME:", time.time() - big_start)



