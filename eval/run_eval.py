import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from tqdm import tqdm
from src.hiver import config
from src.hiver.retriever import Retriever
from src.hiver.drafter import draft_reply
from eval.judge import judge_reply

N = 30


def main():
    df = pd.read_csv(config.GOLDEN_SET)
    df = df[df["label"].notna() & (df["label"] != "")]
    sample = df.sample(n=min(N, len(df)), random_state=42).reset_index(drop=True)
    retriever = Retriever()
    rows = []
    for _, r in tqdm(sample.iterrows(), total=len(sample)):
        hits = retriever.top_k(r["text"], k=3)
        try:
            reply = draft_reply(r["text"], hits, r["label"])
        except Exception as e:
            reply = f"ERROR: {e}"
        scores = judge_reply(r["text"], reply, r["label"])
        rows.append({"tweet_id": r["tweet_id"], "intent": r["label"],
                     "reply": reply, "reply_len": len(reply), **scores})
    out = pd.DataFrame(rows)
    out.to_csv(config.EVAL_RESULTS / "reply_eval.csv", index=False)
    agg = {"n": len(out),
           "mean_helpfulness": float(out["helpfulness"].mean()),
           "mean_tone": float(out["tone"].mean()),
           "mean_accuracy": float(out["accuracy"].mean()),
           "mean_groundedness": float(out["groundedness"].mean()),
           "mean_brevity": float(out["brevity"].mean()),
           "mean_total": float(out["total"].mean()),
           "pct_under_280": float((out["reply_len"] <= 280).mean()),
           "pct_empty": float((out["reply_len"] == 0).mean())}
    with open(config.EVAL_RESULTS / "reply_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()