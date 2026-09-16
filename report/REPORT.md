# Hiver AI Support Agent — Engineering & Evaluation Report

**Project**: AppleSupport AI Support Agent on Twitter  
**Author**: SDE Intern Candidate  
**Date**: September 2024  

---

## 1. Executive Summary & Problem Overview

Customer support on public social media (such as AppleSupport on Twitter) presents a dual challenge: high inbound volume with repetitive informational queries alongside high-risk complaints involving billing disputes, security/account lockouts, and legal threats. An autonomous support agent must:

1. Accurately identify customer intent across a grounded taxonomy.
2. Ground drafted responses on historical, resolved brand interactions.
3. Decisively route queries between automated resolution (`auto_handle`) and human escalation (`escalate`), stating a transparent, policy-based rationale.

Our system achieves **80.5% accuracy (0.673 macro-F1)** on a 200-sample hand-labelled golden set using Groq-hosted LLM classification, outperforming TF-IDF + Logistic Regression (+13.8% accuracy) and naive keyword matching (2.1× lift).

---

## 2. Architecture & System Pipeline

```
Inbound Tweet
     │
     ▼
[Intent Classifier] (LLM / TF-IDF / Keyword)
     │
     ├──► [Semantic Retriever] (all-MiniLM-L6-v2, top-3 similar threads)
     │         │
     │         ▼
     │    [Response Drafter] (Grounded, ≤280 chars, DM steering for sensitive info)
     │
     ▼
[Policy Router] (Word-boundary regex keyword gate, confidence check, intent policy)
     │
     ├── auto_handle  (escalate_to: "ai")
     └── escalate     (escalate_to: "human_dm" / "human_agent" / "human_triage")
```

---

## 3. Intent Taxonomy & Golden Set

The taxonomy covers 6 core operational intents plus an unclassifiable catch-all:
- `order_status`: Inquiries on shipping, tracking, delivery delays.
- `refund_billing`: Inquiries on unauthorized charges, refund requests, subscription costs.
- `account_access`: Inquiries regarding Apple ID lockouts, 2FA, password resets, iCloud authentication.
- `product_issue`: Hardware defects, battery drain, software glitches, iOS crashes.
- `complaint_escalation`: High anger, repeat failed contacts, threats of legal action or churn.
- `general_inquiry`: Feature questions, store hours, compatibility inquiries.
- `unclassifiable`: Spam, gibberish, or text devoid of actionable support context.

The Golden Set consists of 200 hand-labelled tweets sampled from 20,000 AppleSupport inbound tweets, stratified by keyword buckets to ensure balanced representation across all classes.

---

## 4. Evaluation Results

### Intent Classification Benchmark (N=200)

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Trivial Floor (always `general_inquiry`) | 0.155 | 0.038 | Baseline floor |
| Keyword Match | 0.430 | 0.348 | Hand-crafted keyword heuristics |
| TF-IDF + Logistic Regression | 0.667 | 0.470 | 70/30 train/eval split |
| **LLM (Groq `openai/gpt-oss-120b`)** | **0.805** | **0.673** | **Production headline** |

### Reply Quality (LLM-as-Judge, N=30, 0–5 per criterion)

| Criterion | Score | Evaluation Rubric |
|---|---|---|
| Helpfulness | 1.60 | Actionability of the suggestion |
| Tone | 2.07 | Professionalism, empathy, and brand voice |
| Accuracy | 2.13 | Factual consistency with Apple policies |
| Groundedness | 1.73 | Alignment with retrieved historical threads |
| Brevity | 2.67 | Strict compliance with Twitter 280-char limit |
| **Total (out of 25)** | **10.2** | Overall score |

### Routing (audited, not accuracy-scored)

No ground-truth routing labels exist in the dataset. We audited policy consistency across the golden set instead:

| Intent | Auto-handle | Escalate | Escalate Target | Reason |
|---|---|---|---|---|
| `order_status` | ✅ | — | `ai` | Public resolution possible via public tracking portals |
| `general_inquiry` | ✅ | — | `ai` | Public resolution possible (how-to, feature compatibility) |
| `product_issue` | — | ✅ | `human_dm` | Device issue — diagnostics exceed public reply scope. Route to DM |
| `refund_billing` | — | ✅ | `human_dm` | Money/refund issue — needs private billing info and a human commitment |
| `account_access` | — | ✅ | `human_dm` | Account/identity issue — public resolution would require credentials |
| `complaint_escalation` | — | ✅ | `human_agent` | Escalation signal (anger, repeat contact, legal) — human intervention required |
| `unclassifiable` | — | ✅ | `human_triage` | Intent could not be determined confidently — route to human for triage |

---

## 5. Failure Analysis & Diagnostics

1. **Keyword Substring False Positives**:
   - *Observation*: The keyword `"sue"` matched as a naive substring inside legitimate words like `"issue"` or `"tissue"`, incorrectly escalating harmless queries such as *"battery issue on iPhone 17 Pro"* to human legal agents.
   - *Root Cause*: Naive substring matching (`kw in text.lower()`).
   - *Fix*: Upgraded to regex word-boundary matching (`re.search(r"\b" + re.escape(kw) + r"\b", text)`).

2. **Empty LLM Generation (Intermittent)**:
   - *Observation*: Groq's `openai/gpt-oss-120b` reasoning model intermittently consumed token budgets in hidden chain-of-thought, returning an empty `content` field.
   - *Fix*: Switched drafting pipeline to `chat_fast` and raised `max_tokens` to 800.

3. **Retriever Boilerplate Drift**:
   - *Observation*: AppleSupport's real-world Twitter data consists of 86% *"Please DM us"* boilerplate responses. The retriever accurately retrieved these, which constrained reply helpfulness scores. This is faithful to brand behavior rather than a defect.

---

## 6. Safety & Security Directives

For sensitive intents such as `account_access`, strict safety guardrails are codified in `src/hiver/drafter.py`:
> *"If the intent is account_access, NEVER ask for a password, 2FA code, or recovery key. Only ask the customer to DM — the human agent will collect credentials through a secure channel."*

Public tweets must never solicit credentials or personal identification data.

---

## 7. Decision Log

| # | Decision | Rationale |
|---|---|---|
| 1 | **Brand = AppleSupport** | Highest volume in TWCS dataset (14%), diverse technical and customer intents, well-defined DM escalation patterns. |
| 2 | **6 Intents + Unclassifiable** | A concise, mutually exclusive taxonomy allows high inter-annotator agreement and robust classification. |
| 3 | **MiniLM Semantic Embeddings over TF-IDF** | TF-IDF on short tweets suffered from lexical noise (e.g. matching 'late' to irrelevant vulgar tweets). `all-MiniLM-L6-v2` captures contextual semantics. |
| 4 | **Word-boundary regex for escalation keywords** | Initial substring match caused 'sue' to fire inside 'issue', incorrectly escalating "battery issue on iPhone 17 Pro." Fixed with `\b` boundaries. |
| 5 | **Routing reasons explain policy, not restate rules** | Saying "not in allowlist" is circular and uninformative to reviewers. Replaced with per-intent rationale (security, money, human commitment) so escalation decisions are auditable. |
| 6 | **Explicit `escalate_to` Routing Targets** | Distinguishes whether an escalated issue requires a specialized legal/human agent (`human_agent`), an asynchronous private chat agent (`human_dm`), or general intake triage (`human_triage`). |
| 7 | **Preserve Real-world "DM us" Distribution** | Filtering out "DM us" would misrepresent true AppleSupport operations and artificially inflate artificial helpfulness scores. |
| 8 | **Local Web UI Architecture** | Module-level singleton `Agent()` in FastAPI prevents 10-second re-embedding penalties per request, with pure vanilla JS/CSS for zero frontend bloat. |

---

## 8. What I'd Do With One More Week

1. **Multi-turn Context**: Extend classification and drafting from single-tweet analysis to full multi-turn conversation trees.
2. **Learned Escalation Gate**: Train a dedicated classifier for escalation urgency rather than relying on keyword lists.
3. **Cross-Encoder Reranking**: Implement a cross-encoder to rerank the top-20 MiniLM results down to the top-3 most actionable threads.
4. **Automated Human-in-the-Loop Feedback**: Enable customer service managers to label routing decisions and provide direct feedback into the golden set.
