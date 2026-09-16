"""Semantic retrieval of historically resolved AppleSupport threads."""
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config

_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean(text: str) -> str:
    t = str(text)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


class Retriever:
    def __init__(self, df: pd.DataFrame = None, max_pairs: int = 25000, cache: bool = True):
        if df is None:
            df = pd.read_parquet(config.APPLE_THREADS)

        brand = df[df["author_id"] == config.BRAND][
            ["tweet_id", "text", "response_tweet_id"]
        ].copy()
        brand["response_tweet_id"] = brand["response_tweet_id"].astype(str).str.split()

        reply_by_customer = {}
        for _, row in brand.iterrows():
            for rid in row["response_tweet_id"]:
                if rid and rid != "nan":
                    reply_by_customer[rid] = row["text"]

        cust = df[df["author_id"] != config.BRAND][["tweet_id", "text"]].copy()
        cust["brand_reply"] = cust["tweet_id"].map(reply_by_customer)

        pairs = cust.dropna(subset=["brand_reply"]).copy()
        pairs["customer_clean"] = pairs["text"].map(clean)
        pairs["reply_clean"] = pairs["brand_reply"].map(clean)

        pairs = pairs[pairs["customer_clean"].str.len() > 20]
        pairs = pairs[pairs["reply_clean"].str.len() > 15]
        pairs = pairs.drop_duplicates(subset="customer_clean")

        if len(pairs) > max_pairs:
            pairs = pairs.sample(max_pairs, random_state=config.RANDOM_SEED)

        self.pairs = pairs.reset_index(drop=True)

        self.model = SentenceTransformer(_MODEL_NAME)

        cache_path = config.DATA / "retriever_embeddings.npy"
        need_build = True
        if cache and cache_path.exists():
            emb = np.load(cache_path)
            if emb.shape[0] == len(self.pairs):
                self.embeddings = emb
                need_build = False

        if need_build:
            self.embeddings = self.model.encode(
                self.pairs["customer_clean"].tolist(),
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            np.save(cache_path, self.embeddings)

    def top_k(self, query: str, k: int = 3, min_score: float = 0.45) -> list[dict]:
        q = self.model.encode([clean(query)], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        idx = sims.argsort()[::-1][:k]
        out = []
        for i in idx:
            if sims[i] < min_score:
                continue
            out.append({
                "customer": self.pairs.iloc[i]["customer_clean"],
                "reply": self.pairs.iloc[i]["reply_clean"],
                "score": float(sims[i]),
            })
        return out