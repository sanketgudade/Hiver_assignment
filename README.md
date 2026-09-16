# Hiver_Intern_proj — AppleSupport AI Support Agent

An end-to-end AI customer-support agent for **AppleSupport** on Twitter, built for the Hiver SDE Intern take-home. Given an inbound customer tweet, the agent:

1. **Classifies intent** into one of 6 categories (+ `unclassifiable`)
2. **Retrieves** historically similar resolved threads
3. **Drafts** a grounded, on-brand reply (≤280 chars)
4. **Routes** the message as `auto_handle` or `escalate` with a stated reason and handler

**Headline result:** 80.5% intent accuracy / 0.673 macro-F1 on a 200-example hand-labelled golden set — a **2.1× lift** over keyword matching and **+13.8 pts** over TF-IDF.

---

## Table of contents
- [Results](#results)
- [Repo structure](#repo-structure)
- [Setup](#setup)
- [Reproduce headline results in <15 min](#reproduce-headline-results-in-15-min)
- [Using the agent](#using-the-agent)
- [Web UI](#web-ui)
- [Testing](#testing)
- [Data pipeline](#data-pipeline)
- [Design decisions](#design-decisions)
- [Known limitations & failure analysis](#known-limitations--failure-analysis)
- [What I'd do with one more week](#what-id-do-with-one-more-week)
- [Attribution](#attribution)

---

## Results

### Intent classification (N=200 golden set)

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Trivial (always `general_inquiry`) | 0.155 | 0.038 | Baseline floor |
| Keyword match | 0.430 | 0.348 | Hand-written seed keywords |
| TF-IDF + LogisticRegression | 0.667 | 0.470 | 70/30 holdout, trained on golden labels |
| **LLM (Groq `openai/gpt-oss-120b`)** | **0.805** | **0.673** | Headline |

### Reply quality (LLM-as-judge, N=30, 0–5 per criterion)

| Criterion | Run 1 | Run 2 |
|---|---|---|
| Helpfulness | 1.60 | 1.13 |
| Tone | 2.07 | 1.30 |
| Accuracy | 2.13 | 1.40 |
| Groundedness | 1.73 | 1.10 |
| Brevity | 2.67 | 1.83 |
| **Total (out of 25)** | **10.2** | **6.8** |
| % under 280 chars | 100% | 100% |
| % empty replies | 16.7% | 13.3% |

**Run-to-run variance is large (±1.7 pts on total, N=30).** Report as a range, not a point estimate.

### Judge-human agreement (N=20)

| Metric | Value |
|---|---|
| Mean human helpfulness | 2.95 |
| Mean judge helpfulness | 2.60 |
| Quadratic Cohen's kappa | **-0.177** (poor, below chance) |
| Pearson r | -0.236 |
| Mean absolute error | 1.75 |

**The LLM judge disagrees with a human scorer worse than chance.** Every reply-quality number above derives from this unvalidated judge and should be read as a judge-internal signal, not a human-aligned quality measurement. This is the honest check the assignment asked for — and on this task, the judge failed validation.

Plausible causes: (a) the 5-criterion rubric asks the judge to weigh dimensions the human was not scoring (accuracy, tone, groundedness, brevity) while the human only scored helpfulness; (b) N=20 is small and human scoring was done in a single sitting; (c) the drafting model intermittently emits empty replies that the judge and human score differently.

### Golden set

- **200 hand-labelled examples** in `data/golden_set.csv`
- Sampled from 20k filtered AppleSupport inbound mentions (len > 25 chars, deduped)
- Stratified by keyword bucket so all 6 intents are represented
- Distribution: `product_issue` 91, `account_access` 32, `general_inquiry` 31, `complaint_escalation` 25, `order_status` 13, `refund_billing` 7, `unclassifiable` 1

### Routing (audited, not accuracy-scored)

No ground-truth routing labels exist in the dataset — Apple's actual DM/escalate decisions are not observable. We audited policy consistency instead:

| Intent | Decision | Handler | Reason |
|---|---|---|---|
| `order_status` | auto_handle | ai | Public resolution possible |
| `general_inquiry` | auto_handle | ai | Public resolution possible |
| `product_issue` | escalate | human_dm | Diagnostics exceed public reply scope |
| `refund_billing` | escalate | human_dm | Private billing + human commitment |
| `account_access` | escalate | human_dm | Credentials cannot be handled publicly |
| `complaint_escalation` | escalate | human_agent | Human intervention required |

Escalation reasons explain *policy* (security, money, commitment), not just rule restatement.

---

## Repo structure

```
Hiver_Intern_proj/
├── README.md                       # this file
├── requirements.txt                # pinned deps
├── .env                            # GROQ_API_KEY (gitignored)
├── .gitignore
│
├── data/
│   ├── raw/twcs/twcs.csv           # original Kaggle dump (516MB, gitignored)
│   ├── apple_threads.parquet       # filtered AppleSupport slice (~80MB, gitignored)
│   ├── apple_small.parquet         # 20k inbound sample for dev (gitignored)
│   ├── golden_set.csv              # ★ 200 hand-labelled examples
│   └── retriever_embeddings.npy    # cached MiniLM embeddings (~90MB, gitignored)
│
├── src/hiver/
│   ├── __init__.py
│   ├── config.py                   # paths, brand, model names, thresholds
│   ├── intents.py                  # 6-intent taxonomy
│   ├── llm.py                      # Groq/OpenAI client (chat + chat_fast)
│   ├── data_loader.py              # chunked CSV → parquet
│   ├── retriever.py                # MiniLM semantic search over past threads
│   ├── classifier.py               # LLM + TF-IDF + keyword classifiers
│   ├── drafter.py                  # reply generation (with PII guard)
│   ├── router.py                   # auto-handle vs escalate rules (word-boundary regex)
│   └── agent.py                    # orchestrator + CLI
│
├── eval/
│   ├── __init__.py
│   ├── build_golden.py             # stratified sampler
│   ├── run_eval.py                 # classifier eval vs 3 baselines
│   ├── judge.py                    # LLM-as-judge rubric
│   ├── run_reply_eval.py           # reply quality eval
│   ├── judge_agreement.py          # human-vs-judge agreement study
│   └── results/
│       ├── metrics.json            # classifier results
│       ├── predictions.csv         # per-row classifier predictions
│       ├── reply_eval.csv          # per-row reply + judge scores
│       ├── reply_metrics.json      # aggregate reply numbers
│       └── judge_agreement.json    # human-judge kappa study
│
├── scripts/
│   ├── _bootstrap.py               # sys.path shim for scripts/
│   └── test_retriever.py           # manual retriever sanity check
│
├── report/
│   └── REPORT.md                   # 6-page write-up
│
└── web/                            # FastAPI local web UI demo
    ├── app.py                      # FastAPI server & singleton Agent
    └── static/                     # Vanilla frontend (HTML, JS, CSS)
```

---

## Setup

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**Windows only — torch must be CPU-only** or it fails to load `c10.dll`. Install CPU torch *before* `requirements.txt` so `sentence-transformers` picks it up instead of pulling a CUDA build.

If torch still fails to import: install the [Microsoft Visual C++ Redistributable](https://aka.ms/vc14/vc_redist.x64.exe) and reboot.

### 2. API key

Get a free Groq key at https://console.groq.com/keys, then create `.env`:

```
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=
```

The code prefers Groq; OpenAI is a fallback if `GROQ_API_KEY` is missing.

### 3. Data

**Option A — Kaggle CLI:**
```powershell
cd data\raw
kaggle datasets download -d thoughtvector/customer-support-on-twitter
Expand-Archive .\customer-support-on-twitter.zip -DestinationPath . -Force
# You should now have data\raw\twcs\twcs.csv (~516MB)
```

**Option B — HuggingFace mirror (no Kaggle account):**
```powershell
python -c "from datasets import load_dataset; ds = load_dataset('thishankgulati/customer-support-on-twitter', split='train'); ds.to_csv('data/raw/twcs.csv', index=False)"
```

---

## Reproduce headline results in <15 min

Run these four commands in order from the repo root:

```powershell
# 1. Filter AppleSupport threads (~3 min) → apple_threads.parquet + apple_small.parquet
python -m src.hiver.data_loader

# 2. Verify the agent works end-to-end (~15s first run: downloads MiniLM, encodes 22k pairs)
python -m src.hiver.agent --text "My iPhone 15 order is late."

# 3. Classifier eval vs 3 baselines (~3 min, 200 LLM calls)
python eval\run_eval.py
#   → prints: Trivial / Keyword / TF-IDF / LLM accuracies
#   → writes eval\results\metrics.json

# 4. Reply quality eval (~4 min, 30 drafts + 30 judge calls)
python eval\run_reply_eval.py
#   → prints: mean scores per criterion
#   → writes eval\results\reply_metrics.json
```

**Expected outputs:**
- Step 3: `LLM: acc=0.805 macro_f1=0.673`
- Step 4: `mean_total ≈ 6.8–10.2` (varies run to run)

---

## Using the agent

### Single message (CLI)

```powershell
python -m src.hiver.agent --text "This is the THIRD time I'm tweeting you about my locked Apple ID. I want a refund and I'm about to contact my lawyer."
```

Output:
```
=== INPUT ===
...

=== INTENT ===
complaint_escalation  (confidence=0.96)
reason: User angry, repeated contact, threatens legal action.

=== ROUTING ===
ESCALATE  — Contains sensitive keyword 'refund' — requires human.

=== RETRIEVED (3) ===
  [0.69] My Apple ID it says it was locked...
        -> Hi there! Which account are you locked out of...

=== DRAFT REPLY ===
...
```

### Programmatic use

```python
from src.hiver.agent import Agent

agent = Agent()
result = agent.run("My iPhone 15 order is late.")
print(result["intent"])                # "order_status"
print(result["reply"])                 # grounded draft
print(result["routing"]["decision"])   # "auto_handle"
print(result["routing"]["reason"])     # stated reason
print(result["routing"]["handler"])    # "ai" | "human_dm" | "human_agent"
```

### Routing policy

**Auto-handle** only when all of these hold:
- Intent ∈ {`order_status`, `general_inquiry`}
- Classifier confidence ≥ 0.75
- No escalation keyword in text (word-boundary matched)

**Escalate** otherwise, with a stated reason and a target handler:
- Sensitive keyword (`refund`, `lawyer`, `hacked`, `locked out`, ...) → `human_agent`
- Account/billing intent → `human_dm`
- Low confidence → `human_triage`

---

## Web UI

A minimal FastAPI + vanilla JS browser interface for testing the agent locally.

```powershell
cd C:\Users\SANKET\Desktop\Hiver_Final_Submission\Hiver_Intern_proj
.\.venv\Scripts\Activate.ps1
uvicorn web.app:app --reload --port 8000
```

Open **http://127.0.0.1:8000**

**Notes:**
- The first request takes ~10–15s while the retriever model loads.
- Later requests are ~3s each.
- The web UI is read-only against the agent — it does not modify the pipeline.

---

## Testing

### 1. Retriever sanity check

```powershell
python scripts\test_retriever.py
```

Expects: `pairs: 22240`, and top-3 retrieved rows about orders / accounts / hardware / billing for the four queries.

### 2. Agent smoke tests

```powershell
python -m src.hiver.agent --text "My iPhone 15 order is late."
python -m src.hiver.agent --text "I was charged twice for the same app."
python -m src.hiver.agent --text "I'm locked out of my Apple ID."
python -m src.hiver.agent --text "This is the THIRD time. I want a refund and I'm contacting my lawyer."
```

Expected intents: `order_status`, `refund_billing`, `account_access`, `complaint_escalation`.

### 3. Classifier eval

```powershell
python eval\run_eval.py
```

Sanity check: LLM accuracy must exceed TF-IDF. If TF-IDF > LLM, something regressed.

### 4. Reply eval

```powershell
python eval\run_reply_eval.py
```

Sanity check: `pct_under_280` should be ~1.0. `pct_empty` should be < 0.2.

### 5. LLM connectivity

```powershell
python -c "from src.hiver.llm import chat; print(chat('You are a test.', 'Say OK.'))"
```

Should print `OK`.

---

## Data pipeline

### Why not load the full CSV?

`twcs.csv` is 516MB / ~3M rows / ~3–4GB in memory. We never load it fully:

1. **Pass 1** — stream in 100k-row chunks, collect AppleSupport tweet IDs (106,860)
2. **Pass 2** — stream again, keep rows authored by AppleSupport or replying to AppleSupport (143,518 kept)
3. **Write** — save to `apple_threads.parquet` (~80MB)
4. **Sample** — filter to inbound customer tweets (len > 25, contains letters, deduped), sample 20k → `apple_small.parquet`

Peak RAM ~300MB. Full file is streamed twice and never held.

### Why sentence-transformers, not TF-IDF?

TF-IDF similarity on Twitter text is dominated by `@AppleSupport` mentions and stopwords. Top-1 for *"my iPhone order is late"* was *"it's too late they have his midget porn now"* — a lexical match on "late" alone. We switched to `all-MiniLM-L6-v2` (90MB, CPU-friendly), which produces cosines of 0.6+ for semantically related tweets.

### Why 86% of AppleSupport replies are "DM us"

We measured it: 19,149 of 22,240 brand replies contain the phrase "DM us" or a close variant. **This is what AppleSupport actually does on Twitter.** We kept it rather than filtering, because:

- It's the true behavior of the brand we're modeling
- Filtering it would be dishonest ("look, our bot helps more than Apple does!")
- It explains *why* reply helpfulness is capped at 1.6/5 — the corpus teaches boilerplate

---

## Design decisions

Full decision log in `report/REPORT.md` section 7. Summary:

| # | Decision | Why |
|---|---|---|
| 1 | Brand = AppleSupport | Highest volume (14% of dataset), most intent variety, clear DM-escalation pattern |
| 2 | 6 intents + unclassifiable | Small taxonomy is testable; 20+ intents would have <5 examples each |
| 3 | TF-IDF → MiniLM embeddings | Lexical similarity was dominated by @mention noise |
| 4 | Kept "DM us" boilerplate | It's AppleSupport's real behavior — filter and you fake the brand |
| 5 | Class-balanced golden set | Rare intents (`complaint_escalation`) would otherwise have ~2 examples |
| 6 | Keyword-gated escalation | Only 25 `complaint_escalation` examples — too few for a learned gate |
| 7 | Chunked CSV read | 516MB file never sits in RAM |
| 8 | Groq over OpenAI | Free tier, faster, 70B-class reasoning sufficient |
| 9 | `chat_fast` for drafting | `gpt-oss-120b` is a reasoning model; non-reasoning path avoids empty `content` |
| 10 | `max_tokens=800` for drafting | Reasoning models burn tokens internally before emitting text |
| 11 | `min_score=0.45` on retriever | Below this, retrieved examples are noise and hurt drafts |
| 12 | Split eval: classifier vs reply | One "does it work" number hides the drafter weakness |
| 13 | Kept 16.7% empty-reply rate visible | The number is the proof; filtering failures hides the bug |
| 14 | CPU-only torch | Windows CUDA runtime DLL failed to load; CPU is fast enough |
| 15 | Word-boundary regex for escalation keywords | Substring match caused `'sue'` to fire inside `'issue'`, incorrectly escalating a battery complaint. Fixed with `\b` boundaries. |
| 16 | Routing reasons explain policy, not restate rules | "Not in allowlist" is circular. Replaced with per-intent rationale (security / money / commitment). |
| 17 | Hard-coded PII guard in drafter prompt | Even though `account_access` routes to human DM, the drafter prompt also forbids requesting passwords, 2FA codes, or recovery keys — defense in depth. |
| 18 | Ran judge-human agreement despite expecting a positive result | Found κ=-0.177 (worse than chance). Reported it honestly and qualified every downstream reply-quality number. |

---

## Known limitations & failure analysis

1. **Empty replies (10–17%).** The drafting model (`openai/gpt-oss-120b` on Groq) intermittently returns empty `content` for reasoning-heavy prompts. Highest-leverage fix: retry on empty + template fallback.

2. **Reply quality is weak (10.2 / 25).** The retriever surfaces Apple's 86%-boilerplate corpus, so the drafter learns to say "DM us" instead of resolving. This is a data property, not a prompt bug.

3. **Judge-human agreement is poor (κ = -0.177).** The LLM judge disagrees with a human scorer worse than chance. Every reply-quality number in this repo derives from that unvalidated judge and should be read as a judge-internal signal, not a human-aligned quality measure.

4. **Class-balanced golden set is not production-representative.** Real AppleSupport traffic is ~60% `product_issue`; our eval over-represents rare intents to allow per-class measurement.

5. **Keyword substring false positives (addressed).** `"sue"` matched inside `"issue"`, causing incorrect escalation on a battery complaint. Resolved with word-boundary regex.

6. **Single-turn only.** Threads are typically 3+ turns; the agent sees one message.

7. **N=30 for reply eval → wide confidence intervals (±1.7 pts run-to-run).**

8. **N=20 for judge-human agreement → also wide CIs.** A proper study would use 100+ items and multiple raters.

---

## What I'd do with one more week

1. **Fix empty replies** — retry on empty, fall back to a template.
2. **Improve judge validity** — current judge has κ=-0.177 vs human. Options: rewrite the rubric to match what humans actually weight, use a multi-model judge ensemble, or drop LLM-as-judge in favor of a small human panel with clear scoring criteria.
3. **Hand-label 800–1000 more examples** — reduces variance on rare intents.
4. **Learned escalation classifier** — replace keyword gating.
5. **Multi-turn context** — feed the last 3 turns into the classifier and drafter.
6. **Better retriever** — cross-encoder reranking of top-20 to top-3.
7. **Per-intent reply templates with slot filling** — backed off to free-form LLM when slots are missing. Directly addresses the boilerplate issue.
8. **Production monitoring** — log confidence distribution, route the bottom decile to human review, track override rate.

---

## Attribution

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (ThoughtVector, CC-BY-NC-SA-4.0)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0)
- LLM: Groq API (`openai/gpt-oss-120b`)
- Reference reading: scikit-learn TF-IDF guides, sentence-transformers docs, Groq deprecation notices

---

## License

Submission for the Hiver SDE Intern take-home. Not for redistribution.