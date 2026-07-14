from datasets import load_dataset


ds = load_dataset(
    "openbmb/Ultra-FineWeb",
    streaming=True
)

print(ds)
