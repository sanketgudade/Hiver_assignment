import _bootstrap  # noqa
from src.hiver.retriever import Retriever

r = Retriever()
print("pairs:", len(r.pairs))

queries = [
    "my iPhone order is late",
    "locked out of my Apple ID",
    "iPad screen is cracked",
    "I was charged twice for the same app",
]

for q in queries:
    print()
    print("QUERY:", q)
    for ex in r.top_k(q, k=3):
        print(f"  [{ex['score']:.3f}] C: {ex['customer'][:110]}")
        print(f"           R: {ex['reply'][:110]}")