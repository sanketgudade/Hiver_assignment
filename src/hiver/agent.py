"""Orchestrator: text → {intent, reply, routing}."""
import argparse
from .classifier import classify_llm
from .retriever import Retriever
from .drafter import draft_reply
from .router import route


class Agent:
    def __init__(self):
        self.retriever = Retriever()

    def run(self, text: str) -> dict:
        cls = classify_llm(text)
        retrieved = self.retriever.top_k(text, k=3)
        reply = draft_reply(text, retrieved, cls["intent"])
        routing = route(cls["intent"], cls["confidence"], text)
        return {
            "input": text,
            "intent": cls["intent"],
            "confidence": cls["confidence"],
            "classifier_reason": cls["reason"],
            "reply": reply,
            "routing": routing,
            "retrieved": retrieved,
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    agent = Agent()
    out = agent.run(args.text)

    print("\n=== INPUT ===")
    print(out["input"])
    print("\n=== INTENT ===")
    print(f"{out['intent']}  (confidence={out['confidence']:.2f})")
    print(f"reason: {out['classifier_reason']}")
    print("\n=== ROUTING ===")
    print(f"{out['routing']['decision'].upper()}  — {out['routing']['reason']}")
    print(f"\n=== RETRIEVED ({len(out['retrieved'])}) ===")
    for r in out["retrieved"]:
        print(f"  [{r['score']:.2f}] {r['customer'][:90]}")
        print(f"        -> {r['reply'][:90]}")
    print("\n=== DRAFT REPLY ===")
    print(out["reply"])
    print()