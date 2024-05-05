# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch.multiprocessing as mp
import time

if __name__ == '__main__':
    mp.set_start_method('spawn')
# mp.set_start_method('forkserver', force=True)


from data import select_token_range 
from generate_data import generate_data

import torch
import boto3



from transformer_lens import HookedTransformer

from attention_attribution import (
    get_bos_ablate_for_head,
)

from d_types import DTYPES

def data_generator(rank, num_devices):
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
        aws_secret_access_key='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',
        region_name="us-east-2",
    )
    s3 = session.client("s3")

    generate_data(
        s3_client=s3,
        bos_ablate_for_head=bos_ablate_for_head,
        num_threads=num_devices,
        thread_id=rank,
        seq_batch_size=3,
        num_batches=1,
        device=f"cuda:{rank}",
        chunk_size=10
    )


# %%
if __name__ == '__main__':
    big_start = time.time()
    # mp.set_start_method('spawn')
    # num_devices = torch.cuda.device_count()
    num_devices = 1

    # %%
    processes = []

    # multiprocessing.set_start_method('spawn', force=True)
    # mp.set_start_method('spawn')

    for i in range(num_devices):
        # def data_generator():
        #     session = boto3.Session(
        #         aws_access_key_id='AKIA2JACKDS2BWWPSBK5',
        #         aws_secret_access_key='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',
        #         region_name="us-east-2",
        #     )
        #     s3 = session.client("s3")

        #     generate_data(
        #         s3_client=s3,
        #         bos_ablate_for_head=bos_ablate_for_head,
        #         num_threads=num_devices,
        #         thread_id=i,
        #         seq_batch_size=3,
        #         num_batches=1,
        #         device=f"cuda:{i}",
        #         chunk_size=20
        #     )


        # data_generator = partial(
        #     generate_data,
        #     s3_client=s3,
        #     bos_ablate_for_head=bos_ablate_for_head,
        #     num_threads=num_devices,
        #     thread_id=i,
        #     seq_batch_size=3,
        #     num_batches=1,
        #     device=f"cuda:{i}",
        #     chunk_size=20
        # )

        p = mp.Process(target=data_generator, args=(i, num_devices))
        p.start()

        processes.append(p)

    for p in processes:
        p.join()

    print()
    print("TOTAL TIME:", time.time() - big_start)

