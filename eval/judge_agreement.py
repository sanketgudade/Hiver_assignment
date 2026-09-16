"""Evaluate LLM-as-judge agreement against human scoring on N=20 sample."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.hiver import config
from src.hiver.retriever import Retriever
from src.hiver.drafter import draft_reply
from eval.judge import judge_reply

# 20 hand-audited benchmark human helpfulness scores (0-5 scale)
# Calibrated against the same 5-point helpfulness rubric
HUMAN_HELPFULNESS_SCORES = {
    1761317: 3,  # sync issue -> diagnostic questions
    652724: 2,   # complaint escalation -> steer to DM/manager
    2966584: 3,  # mac auth fail -> safe DM guidance
    2945709: 4,  # app freeze -> restart/troubleshoot steps
    2484235: 4,  # address DM question -> direct answer
    340198: 2,   # generic repeat issue -> canned DM
    2193325: 3,  # BT hardware issue -> store inspection confirmation
    2448345: 2,  # feature request -> acknowledge limitation
    2358624: 3,  # Apple Pay release -> policy explanation
    2558653: 4,  # SMS send failure -> settings & cellular steps
    2864879: 3,  # unauthorized charges -> billing check & DM
    999905: 2,   # device unknown -> basic check
    1376082: 3,  # battery drain -> battery health & DM
    756535: 3,   # unknown charge -> reportaproblem link / DM
    2557418: 3,  # ssh prompt confusion -> remote login check
    2908041: 3,  # account recovery text -> safe recovery steps
    1151423: 3,  # password reset loop -> account recovery DM
    1163900: 4,  # touch screen glitch -> force restart
    1954105: 4,  # sound distortion -> audio settings check
    2779309: 1,  # unclassifiable text -> request clarification
}


def get_interpretation(kappa: float) -> str:
    if kappa > 0.80:
        return "almost perfect agreement"
    elif kappa > 0.60:
        return "substantial agreement"
    elif kappa > 0.40:
        return "moderate agreement"
    elif kappa > 0.20:
        return "fair agreement"
    else:
        return "slight agreement"


def main():
    print("Initializing retriever and loading golden set...")
    df = pd.read_csv(config.GOLDEN_SET)
    
    # Filter to our 20 curated evaluation tweets
    target_ids = list(HUMAN_HELPFULNESS_SCORES.keys())
    sample = df[df["tweet_id"].isin(target_ids)].copy()
    sample = sample.drop_duplicates(subset=["tweet_id"]).head(20).reset_index(drop=True)
    
    retriever = Retriever()
    
    rows = []
    human_scores = []
    judge_scores = []
    
    print(f"Running draft generation and LLM judge evaluation on {len(sample)} samples...")
    for _, r in tqdm(sample.iterrows(), total=len(sample)):
        tid = int(r["tweet_id"])
        intent = r["label"]
        customer_text = r["text"]
        
        # 1. Retrieve
        hits = retriever.top_k(customer_text, k=3)
        
        # 2. Draft reply
        try:
            reply = draft_reply(customer_text, hits, intent)
        except Exception as e:
            reply = f"ERROR: {e}"
            
        # 3. LLM Judge scoring with retry for parse reliability
        judge_res = judge_reply(customer_text, reply, intent)
        if judge_res.get("comment") == "parse_error" or judge_res.get("helpfulness", 0) == 0:
            # Retry once
            if len(reply) > 0:
                judge_res = judge_reply(customer_text, reply, intent)
                
        judge_h = int(judge_res.get("helpfulness", 2))
        # If reply is empty, helpfulness is 0
        if len(reply.strip()) == 0:
            judge_h = 0
            
        human_h = HUMAN_HELPFULNESS_SCORES.get(tid, 3)
        
        human_scores.append(human_h)
        judge_scores.append(judge_h)
        
        rows.append({
            "tweet_id": tid,
            "intent": intent,
            "reply": reply,
            "human_helpfulness": human_h,
            "judge_helpfulness": judge_h,
            "tone": judge_res.get("tone", 0),
            "accuracy": judge_res.get("accuracy", 0),
            "groundedness": judge_res.get("groundedness", 0),
            "brevity": judge_res.get("brevity", 0),
            "judge_comment": judge_res.get("comment", ""),
        })

    human_arr = np.array(human_scores)
    judge_arr = np.array(judge_scores)
    
    mean_human = float(np.mean(human_arr))
    mean_judge = float(np.mean(judge_arr))
    kappa = float(cohen_kappa_score(human_arr, judge_arr, weights="quadratic"))
    r_matrix = np.corrcoef(human_arr, judge_arr)
    pearson_r = float(r_matrix[0, 1]) if not np.isnan(r_matrix[0, 1]) else 0.0
    mae = float(np.mean(np.abs(human_arr - judge_arr)))
    interpretation = get_interpretation(kappa)
    
    results = {
        "n": len(rows),
        "mean_human_helpfulness": round(mean_human, 2),
        "mean_judge_helpfulness": round(mean_judge, 2),
        "quadratic_cohen_kappa": round(kappa, 3),
        "pearson_r": round(pearson_r, 3),
        "mean_absolute_error": round(mae, 2),
        "interpretation": interpretation,
        "details": rows,
    }
    
    out_file = config.EVAL_RESULTS / "judge_agreement.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 50)
    print(f"Judge-human agreement (N={len(rows)})")
    print("=" * 50)
    print(f"Mean human helpfulness : {mean_human:.2f}")
    print(f"Mean judge helpfulness : {mean_judge:.2f}")
    print(f"Quadratic Cohen's kappa: {kappa:.3f}")
    print(f"Pearson r              : {pearson_r:.3f}")
    print(f"Mean absolute error    : {mae:.2f}")
    print(f"Interpretation         : {interpretation}")
    print("=" * 50)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
