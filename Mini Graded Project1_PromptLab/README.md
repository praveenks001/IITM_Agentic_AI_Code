# MP1 · Prompt Lab — Compare LLM Strategies on a Task

> *Mini Project 1 · Week 5 deliverable · Foundations phase closing.
> Doable in 3 days. Revision of W1-W5 skills via one integrative build.*

---

## What this package contains

### For learners

| File | Purpose |
|---|---|
| `learner/MP1_Brief.md` | Project brief — what to build, in what shape, what to submit |
| `learner/MP1_Starter_Template.ipynb` | Notebook skeleton with TODOs |
| `learner/MP1_FAQ.md` | Anticipated questions + answers |
| `data/job_snippets.jsonl` | 10 job postings (the task input) |
| `data/golden_set.jsonl` | Reference extractions for scoring |
| `requirements.txt` | Python dependencies |

### For instructors / graders

| File | Purpose |
|---|---|
| `instructor/MP1_Rubric.md` | Scoring rubric (1-4 scale, four dimensions) |
| `instructor/MP1_Solution_Document.md` | Reference solution walk-through + expected numbers + what to look for |
| `instructor/MP1_Reference_Solution.ipynb` | Complete working solution notebook |
| `instructor/MP1_Reference_Solution.py` | Same as runnable script (for CI spot-checks) |
| `instructor/MP1_Sample_Submission.md` | What a strong learner's writeup looks like |

---

## At a glance

- **Project:** Compare 4 prompting strategies (zero-shot, few-shot, structured, chain-of-thought) on a structured extraction task.
- **Task:** Extract `company`, `role`, and `years_experience_required` from a job posting snippet.
- **Inputs:** 10 job snippets (provided)
- **Golden set:** 10 reference extractions (provided)
- **Effort:** ~5-8 hours over 3 days
- **Spend:** ~$0.05 to $0.20 on OpenAI API calls
- **Submission:** Notebook + 1-page writeup

---

## Where this fits in the curriculum

MP1 sits at the **closing of Phase 1 (Foundations)** — the same week as M1 / DR #1.

It's deliberately small. The point isn't to test new skills; it's to give learners one integrative build that touches everything they learned in W1-W5:

| W1-W5 skill | How MP1 exercises it |
|---|---|
| W1 — System prompt leverage | The 4 strategies *are* 4 different prompting choices |
| W2 — Async batch pipeline | Run 10 inputs × 4 strategies = 40 calls in parallel |
| W3 — Clean code structure | Single notebook with logical section breaks |
| W4 — Cost + latency tracking | Each strategy reports cost/latency for comparison |
| W5 — Golden set + LLM judge | The scoring step uses the W5 judge pattern |

No new concepts. No new tools. Just revision via integration.

---

## How to grade

Open `instructor/MP1_Solution_Document.md` first — it walks through what a strong submission looks like, with expected numbers and common mistakes to watch for.

Then use `instructor/MP1_Rubric.md` for the actual scoring. Four dimensions (Completion · Correctness · Reflection · Code Quality), each 1-4. Generous rubric — effort and clarity over perfection.

---

## Where this lives in the cohort's repo

```
capstone-repo/
├── (W1-W5 lab work)
├── mp1/
│   ├── mp1_prompt_lab.ipynb       # or .py
│   ├── data/                       # symlink or copy of provided data
│   └── mp1_writeup.md              # the 1-page reflection
└── docs/
    └── adr/0002-prompting-strategy.md   # optional ADR if they want one
```

---

## Constraints (locked by the curriculum)

- **Vocareum-compatible** — runs in the cohort lab environment
- **OpenAI API only** — no Anthropic, no local models in MP1
- **Default model: gpt-4o-mini** — `gpt-4o` used only for the LLM-as-judge step
- **British spelling** throughout cohort-facing docs

---

*Six mini projects total across the 30-week programme. MP1 sits here, in W5.*
