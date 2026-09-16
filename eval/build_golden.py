"""Stratified sampler for golden evaluation set. Priority-ordered bucketing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.hiver import config

BUCKETS = {
    "complaint_escalation": [
        "third time", "3rd time", "2nd time", "second time", "again and again",
        "manager", "lawyer", "attorney", "sue", "small claims", "bbb",
        "unacceptable", "ridiculous", "terrible", "useless", "worst",
        "never again", "disappointed", "fed up", "report you", "escalate",
        "horrible", "disgusting", "furious", "angry",
    ],
    "refund_billing": [
        "refund", "charge", "billing", "invoice", "receipt",
        "subscription", "double charge", "overcharged", "money back",
    ],
    "account_access": [
        "login", "sign in", "password", "locked out", "apple id",
        "verification", "two-factor", "2fa", "sign-in", "account locked",
    ],
    "order_status": [
        "order", "shipping", "delivery", "tracking", "where is my",
        "when will", "hasn't arrived", "not arrived", "shipped",
    ],
    "product_issue": [
        "not working", "broken", "crash", "won't turn on", "stuck",
        "frozen", "screen", "battery", "keeps restarting", "error",
        "bug", "glitch", "not charging",
    ],
    "general_inquiry": [
        "how do i", "does it", "can i", "when is", "what is",
        "is there a way", "how to",
    ],
}

BUCKET_PRIORITY = list(BUCKETS.keys())


def bucket_of(text: str) -> str:
    t = text.lower()
    for b in BUCKET_PRIORITY:
        for kw in BUCKETS[b]:
            if kw in t:
                return b
    return "other"


def main():
    df = pd.read_parquet(config.APPLE_SMALL)
    df["bucket"] = df["text"].map(bucket_of)

    per_bucket = 200 // len(BUCKETS)  # ~33 each
    picks = []
    for b in BUCKET_PRIORITY:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            print(f"WARNING: bucket '{b}' had 0 rows")
            continue
        picks.append(sub.sample(n=min(per_bucket, len(sub)), random_state=42))

    golden = pd.concat(picks).sample(frac=1, random_state=42).reset_index(drop=True)

    # Fill to 200 from 'other' if short
    if len(golden) < 200:
        other = df[df["bucket"] == "other"]
        need = 200 - len(golden)
        extra = other.sample(n=min(need, len(other)), random_state=42)
        extra = extra.assign(bucket="other")
        golden = pd.concat([golden, extra]).sample(frac=1, random_state=42).reset_index(drop=True)

    golden["label"] = ""
    golden["notes"] = ""
    golden[["tweet_id", "text", "bucket", "label", "notes"]].to_csv(
        config.GOLDEN_SET, index=False
    )
    print(f"\nWrote {len(golden)} rows to {config.GOLDEN_SET}")
    print(golden["bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()