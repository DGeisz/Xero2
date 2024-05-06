from data import select_token_range
from d_types import DTYPES

import torch
import pathlib
import time

from transformer_lens import HookedTransformer

from attention_attribution import (
    get_attn_attrib_on_seq,
)


file_template = "attr-type-{}-batch-start-{}-batch-size-{}.pt"


def get_file_name(attr_type: str, batch_i: int, batch_size=125):
    assert attr_type in ("bos", "zero", "mix")

    return file_template.format(attr_type, batch_i, batch_size)


def generate_data(
    s3_client,
    bos_ablate_for_head,
    num_threads=8,
    thread_id=0,
    seq_batch_size=125,
    start_batch=0,
    num_batches=8,
    dtype="bf16",
    device="cuda:0",
    start_token=10,
    chunk_size=10,
):
    print(device)

    cfg = {"model": "gpt2", "device": device, "enc_dtype": dtype}
    bos_ablate_for_head = bos_ablate_for_head.to(device)

    model = (
        HookedTransformer.from_pretrained(cfg["model"])
        .to(DTYPES[cfg["enc_dtype"]])
        .to(cfg["device"])
    )

    start_batch = (start_batch // (num_threads)) * num_threads

    start_batch *= num_threads

    print("start batch", start_batch)

    start_time = time.time()

    for i in range(num_batches):

        b_start_index = start_batch + ((num_threads * i) + thread_id)
        batch_start_index = b_start_index * seq_batch_size

        tokens = select_token_range(batch_start_index, seq_batch_size).to(device)

        N, _ = tokens.shape

        bos_list = []
        zero_list = []
        mix_list = []

        print(
            "Thread:",
            thread_id,
            "Batch:",
            batch_start_index,
            "Tokens:",
            N,
            f"Elapsed: {time.time() - start_time:.2f}",
        )

        i = 0

        for chunk in torch.split(torch.arange(N), chunk_size):
            print(
                "Thread:",
                thread_id,
                "Chunk:",
                i,
                "Batch:",
                batch_start_index,
                f"Elapsed: {time.time() - start_time:.2f}",
            )

            bos, zero, mix = get_attn_attrib_on_seq(
                model, tokens[chunk], start_token, bos_ablate_for_head
            )

            bos_list.append(bos.cpu())
            zero_list.append(zero.cpu())
            mix_list.append(mix.cpu())
            i += 1

        bos = torch.cat(bos_list, dim=0)
        zero = torch.cat(zero_list, dim=0)
        mix = torch.cat(mix_list, dim=0)

        print(
            "-- BEGIN SAVE --" "Thread:",
            thread_id,
            "Chunk:",
            i,
            "Batch:",
            batch_start_index,
            f"Elapsed: {time.time() - start_time:.2f}",
        )

        bos_file_name = file_template.format("bos", b_start_index, seq_batch_size)
        zero_file_name = file_template.format("zero", b_start_index, seq_batch_size)
        mix_file_name = file_template.format("mix", b_start_index, seq_batch_size)

        torch.save(bos, bos_file_name)
        torch.save(zero, zero_file_name)
        torch.save(mix, mix_file_name)

        s3_client.upload_file(
            bos_file_name, "mech-interp", f"attribution/{bos_file_name}"
        )
        s3_client.upload_file(
            zero_file_name, "mech-interp", f"attribution/{zero_file_name}"
        )
        s3_client.upload_file(
            mix_file_name, "mech-interp", f"attribution/{mix_file_name}"
        )

        pathlib.Path.unlink(bos_file_name)  # type: ignore
        pathlib.Path.unlink(zero_file_name)  # type: ignore
        pathlib.Path.unlink(mix_file_name)  # type: ignore

        print(
            "-- END SAVE --" "Thread:",
            thread_id,
            "Chunk:",
            i,
            "Batch:",
            batch_start_index,
            f"Elapsed: {time.time() - start_time:.2f}",
        )

    print(
        "-- COMPLETED RUN -- ",
        "Thread:",
        thread_id,
        f"Elapsed: {time.time() - start_time:.2f}",
    )
