import os
import json
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


OUT_DIR = (
    "data/openbmb_UltraFineWeb_100m_tokens"
)


# 总100M tokens
TRAIN_TOKENS = 90000000
VALID_TOKENS = 10000000


os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/cosmo2-tokenizer"
)


print(
    "vocab size:",
    tokenizer.vocab_size
)


# 创建memmap
train_data = np.memmap(
    os.path.join(
        OUT_DIR,
        "train.bin"
    ),
    dtype=np.uint32,
    mode="w+",
    shape=(TRAIN_TOKENS,)
)


valid_data = np.memmap(
    os.path.join(
        OUT_DIR,
        "valid.bin"
    ),
    dtype=np.uint32,
    mode="w+",
    shape=(VALID_TOKENS,)
)


# streaming读取UltraFineWeb
dataset = load_dataset(
    "openbmb/Ultra-FineWeb",
    split="en",
    streaming=True
)


train_pos = 0
valid_pos = 0


for item in tqdm(dataset):

    # 兼容字段
    text = item["content"]


    ids = tokenizer(
        text,
        add_special_tokens=False
    ).input_ids


    # 写train
    if train_pos < TRAIN_TOKENS:

        n = min(
            len(ids),
            TRAIN_TOKENS - train_pos
        )

        train_data[
            train_pos:train_pos+n
        ] = ids[:n]

        train_pos += n

    # 写valid
    else:

        n = min(
            len(ids),
            VALID_TOKENS - valid_pos
        )

        valid_data[
            valid_pos:valid_pos+n
        ] = ids[:n]

        valid_pos += n


    if (
        train_pos >= TRAIN_TOKENS
        and valid_pos >= VALID_TOKENS
    ):
        break


train_data.flush()
valid_data.flush()


manifest = {
    "train_file": "train.bin",
    "valid_file": "valid.bin",
    "dtype": "uint32",
    "vocab_size": tokenizer.vocab_size,
    "train_tokens": TRAIN_TOKENS,
    "valid_tokens": VALID_TOKENS
}


with open(
    os.path.join(
        OUT_DIR,
        "manifest.json"
    ),
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


import gc

try:
    del dataset
except:
    pass

gc.collect()

print("finished")
