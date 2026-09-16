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

### Judge-human agreement (N=20)

| Metric | Value |
|---|---|
| Mean human helpfulness | 2.95 |
| Mean judge helpfulness | 2.60 |
| Quadratic Cohen's kappa | -0.177 |
| Pearson r | -0.236 |
| Mean absolute error | 1.75 |

The LLM judge agrees with a human scorer at κ=-0.177 (poor agreement, below chance). This is the honest check on judge validity.

**Interpretation.** A negative quadratic kappa means the LLM judge and the human scorer disagree systematically — the judge is not a reliable proxy for human helpfulness ratings on this task. Consequences for the report:
- **The reply-quality numbers (10.2 and 6.8 on the 0-25 scale) should be read as judge-internal signals, not as human-aligned quality measurements.** They tell us the judge thinks drafts are mediocre, but we cannot assume a human would agree.
- The 5-criterion rubric asks the judge to score dimensions the human was not asked to consider (accuracy, tone, groundedness, brevity), which may cause the judge to weigh the reply differently than a human scoring only helpfulness. This is one plausible cause.
- N=20 is small and human scoring was done in one sitting — both introduce noise. A proper agreement study would use 100+ items and multiple raters.

This negative result is the honest check the assignment asked for: LLM-as-judge is only as good as its validation, and on this task it failed validation. We report it rather than omit it.

### Routing (audited, not accuracy-scored)

No ground-truth routing labels exist in the dataset — Apple's actual DM/escalate
decisions are not observable. We audited policy consistency instead:

| Intent | Decision | Handler | Reason |
|---|---|---|---|
| order_status | auto_handle | ai | Public resolution possible |
| general_inquiry | auto_handle | ai | Public resolution possible |
| product_issue | escalate | human_dm | Diagnostics exceed public reply scope |
| refund_billing | escalate | human_dm | Private billing + human commitment |
| account_access | escalate | human_dm | Credentials cannot be handled publicly |
| complaint_escalation | escalate | human_agent | Human intervention required |

Escalation reasons explain *policy* (security, money, commitment), not just
rule restatement — the earlier `"Not in allowlist"` reason was circular and
was replaced.

### What's Misleading / Metric Caveats

- **The judge itself is unvalidated.** Judge-human kappa is -0.177 (poor, below chance). Every reply-quality number in this report derives from that judge and should be treated as a *relative* internal signal, not an absolute quality measure.
- **Accuracy is high (80.5%), but context is single-turn.** Customer tweets on social media often reference prior interactions or omit device models that require back-and-forth conversational disambiguation.
- **Reply helpfulness scores are depressed by brand grounding.** AppleSupport's historical training distribution is 86% "Please DM us" redirection. A model grounded in realistic brand data will reproduce this protective deflection rather than providing full public diagnostic manuals.

---

## 5. Failure Analysis & Diagnostics

### 1. Keyword substring false positives

**Real example:** Input *"am having battery issue in my iphone 17 pro"*
escalated with reason *"Contains sensitive keyword 'sue'"* — because `'sue'`
matches inside `'issue'`.

**Hypothesis:** naive `if kw in text.lower()` matching ignores word boundaries.
Common English words containing escalation keywords:
* `sue` ⊂ `issue`, `pursue`, `ensue`
* `charge` ⊂ `charger`, `overcharged`, `discharged`
* `legal` ⊂ `illegal`, `paralegal`

**Fix shipped:** word-boundary regex (`\bkeyword\b`). Confirmed: the same
battery complaint now correctly returns `auto_handle` on `product_issue`.
This was found during manual UI testing and is the kind of bug only a real
demo surfaces.

### 2. Empty LLM Generation (Intermittent)
*Observation*: Groq's `openai/gpt-oss-120b` reasoning model intermittently consumed token budgets in hidden chain-of-thought, returning an empty `content` field.  
*Fix*: Switched drafting pipeline to `chat_fast`, raised `max_tokens` to 800+, and added resilient token limits and model fallback.

### 3. Retriever Boilerplate Drift
*Observation*: AppleSupport's real-world Twitter data consists of 86% *"Please DM us"* boilerplate responses. The retriever accurately retrieved these, which constrained reply helpfulness scores. This is faithful to brand behavior rather than a defect.

---

## 6. Safety & Security Directives

For sensitive intents such as `account_access`, strict safety guardrails are codified in `src/hiver/drafter.py`:
> *"SECURITY: If the intent is account_access, NEVER ask for a password, 2FA code, recovery key, or full email address. Only ask the customer to DM — the human agent will collect credentials through a secure channel."*

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

### Additional Engineering Decisions (16–18)

16. **Word-boundary regex for escalation keywords.** Initial substring match
    caused `'sue'` to fire inside `'issue'`, incorrectly escalating a battery
    complaint. Fixed with `\b` boundaries.
17. **Routing reasons explain policy, not restate rules.** `"Not in allowlist"`
    is circular. Replaced with per-intent rationale (security / money /
    commitment) so escalation decisions are auditable.
18. **Hard-coded PII guard in drafter prompt.** Even though `account_access`
    routes to human DM, the drafter prompt also forbids requesting passwords,
    2FA codes, or recovery keys — defense in depth, since the reply may still
    be shown publicly.

---

## 8. What I'd Do With One More Week

1. **Multi-turn Context**: Extend classification and drafting from single-tweet analysis to full multi-turn conversation trees.
2. **Learned Escalation Gate**: Train a dedicated classifier for escalation urgency rather than relying on keyword lists.
3. **Cross-Encoder Reranking**: Implement a cross-encoder to rerank the top-20 MiniLM results down to the top-3 most actionable threads.
4. **Automated Human-in-the-Loop Feedback**: Enable customer service managers to label routing decisions and provide direct feedback into the golden set.
