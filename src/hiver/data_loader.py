"""Memory-safe loader. Never reads the 516MB CSV fully into RAM."""
import pandas as pd
from tqdm import tqdm

from . import config


def build_apple_threads(chunksize: int = 100_000) -> pd.DataFrame:
    print("[1/3] Pass 1: collecting AppleSupport tweet ids...")
    apple_ids = set()
    for chunk in tqdm(pd.read_csv(config.RAW, chunksize=chunksize, dtype=str)):
        ids = chunk.loc[chunk["author_id"] == config.BRAND, "tweet_id"].dropna()
        apple_ids.update(ids.tolist())
    print(f"      found {len(apple_ids):,} AppleSupport tweets")

    print("[2/3] Pass 2: filtering rows related to AppleSupport...")
    keep = []
    for chunk in tqdm(pd.read_csv(config.RAW, chunksize=chunksize, dtype=str)):
        mask = (
            (chunk["author_id"] == config.BRAND)
            | (chunk["in_response_to_tweet_id"].isin(apple_ids))
        )
        sub = chunk.loc[mask]
        if not sub.empty:
            keep.append(sub)

    df = pd.concat(keep, ignore_index=True)
    print(f"      kept {len(df):,} rows")

    print("[3/3] Writing parquet...")
    df.to_parquet(config.APPLE_THREADS, index=False)
    print(f"      -> {config.APPLE_THREADS}")
    return df


def load_apple_threads() -> pd.DataFrame:
    return pd.read_parquet(config.APPLE_THREADS)


def build_small_sample(n: int = None) -> pd.DataFrame:
    n = n or config.SMALL_SAMPLE_N
    df = load_apple_threads()

    brand_ids = set(df.loc[df["author_id"] == config.BRAND, "tweet_id"])
    cust = df[df["author_id"] != config.BRAND].copy()
    cust = cust[cust["in_response_to_tweet_id"].isin(brand_ids)]
    cust = cust[cust["text"].astype(str).str.len() > 25]
    cust = cust[cust["text"].astype(str).str.contains("[A-Za-z]{4,}", regex=True, na=False)]
    cust = cust.drop_duplicates(subset="text")

    sample = cust.sample(n=min(n, len(cust)), random_state=config.RANDOM_SEED)
    sample.to_parquet(config.APPLE_SMALL, index=False)
    print(f"Wrote {len(sample):,} rows -> {config.APPLE_SMALL}")
    return sample


if __name__ == "__main__":
    build_apple_threads()
    build_small_sample()