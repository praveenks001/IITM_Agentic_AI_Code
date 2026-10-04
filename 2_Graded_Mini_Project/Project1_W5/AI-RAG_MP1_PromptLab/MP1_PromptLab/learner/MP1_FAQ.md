# MP1 · FAQ

> *Questions learners actually ask. Skim this before starting and
> again if you get stuck. Updated as new questions come up.*

---

## Scope

### "Do I need to use all four strategies?"

Yes. The whole point is the comparison. A 3-strategy submission can
still earn credit but loses marks on Completion.

### "Can I add a fifth strategy?"

Yes, if you want. Common additions: a "minimal" baseline (one
word, no instructions), or a "schema-only" variant. Don't go past
6 — it bloats the comparison table.

### "Should I use my capstone documents instead of job snippets?"

No. Use the provided job snippets. Same task for everyone makes
grading fair. After you submit, if you have time, re-run your
winning strategy against your capstone domain — that's the
*personal calibration* exercise mentioned in the brief.

### "Can I use a different model than gpt-4o-mini?"

No — keep gpt-4o-mini for the four strategies (so the only
variable is the prompt). The judge step uses gpt-4o because we
want the strongest judge available.

If you really want to compare models, do it as a stretch after
the main exercise. Don't mix it in.

### "Can I use tool-calling for the structured strategy?"

Not for MP1. Tool-calling is a real strategy, but it's a different
mechanism — it makes the comparison less clean. Stick to prompt
text + JSON in MP1. Tool-calling shows up properly in MP4.

---

## Data + task

### "The years_experience for j10 is null. Is that a typo?"

No — that's deliberate. j10 says *"we don't list a specific years
requirement"*. The correct extraction is `null`. We're testing
whether your strategy fabricates a number when the data isn't there.

A good strategy will return `null`. A weak prompt will hallucinate
(usually "5" or "3+").

### "What if my parsed JSON has slightly different field names?"

Normalise. The starter assumes the fields are `company`, `role`,
and `years_experience_required`. If your prompt produces
`company_name` or `years_required`, either:

- (a) fix the prompt to use the agreed field names, or
- (b) map them in your parser before scoring.

Either is fine. Just be consistent.

### "What's the right answer for 'around 6 years' or '7+'?"

Take the stated number. *"Around 6 years"* → 6. *"7+ years"* → 7.
*"Between 3 and 5"* → 3 (the minimum). The golden set follows
this convention; your strategy needs to too.

### "What if the model returns '5+'?"

For scoring purposes, `5` and `5+` should both count as a match
against golden `5`. Strip the `+` in your normalisation. (This is
a worthwhile observation for the writeup: *"my few-shot version
returned '5+' until I added a rule to use the integer"*.)

---

## Async + batching

### "Do I really need async? Can I just loop?"

You can loop, but async is the W2 lesson — running 40 calls
sequentially takes ~5 minutes; in parallel it takes ~30 seconds.
The grading rubric considers async usage a quality signal.

If async confuses you, do a sync version first to get the logic
right, then convert. That's a valid workflow.

### "asyncio inside a notebook gives me weird errors"

In Jupyter, use `await` directly at the top level — no
`asyncio.run`. If you see `RuntimeError: This event loop is already
running`, that's the symptom — drop the `asyncio.run` wrapper.

### "Should I rate-limit my calls?"

Not for 40 calls. OpenAI's tier-1 limits are well above what MP1
needs. If you're hitting rate limits, you've accidentally written
a retry loop without a cap.

---

## Scoring

### "Should accuracy be 0/1/2/3 or 0%/33%/67%/100%?"

Either works. The reference solution uses 0-3 (count of correct
fields). When you aggregate to the comparison table, divide by 3
to get a 0.0-1.0 score. Both are equivalent.

### "Should I score by-field instead of aggregate?"

The aggregate score (0-3) is required. By-field is an excellent
optional addition — *"the structured strategy gets `company`
right 100% but `years` only 70%"* is a useful finding for the
writeup. Add it if you want; don't lose marks if you skip it.

### "My LLM judge gives different scores on repeated runs"

That's a real issue. Mitigations:

- Set `temperature=0.0` on the judge call (most important)
- Run each judge call once; don't re-judge unless something changed
- Note this in your writeup — *"the judge had ~1 point of variance
  on repeat runs"* is a real finding

If you want more rigor, run the judge 3 times and take the median.
Optional for MP1.

### "Should I worry if my LLM judge disagrees with my accuracy score?"

Yes, but it's a finding, not a bug. If judge gives a 4 when
accuracy is 2/3, the judge is being generous. If judge gives a 2
when accuracy is 3/3, the judge is finding something the strict
match missed.

Note these disagreements in the writeup. This is the W5 calibration
lesson surfacing in MP1.

---

## Cost + spend

### "How do I track cost?"

For each API call, multiply input/output token counts by the
per-token rates in `RATES` (set up in the starter). Accumulate
into the result row. Sum by strategy at the end.

### "My total spend is $1.20 — is that wrong?"

Probably. Likely causes:

- You used `gpt-4o` instead of `gpt-4o-mini` for the main 40 calls
- You re-ran the whole thing 5+ times while debugging
- You judged each call multiple times

Spot-check: cost per call should be ~$0.0001 (mini) or ~$0.003
(judge). 40 main + 40 judge calls = ~$0.12-0.15 total.

### "Can I cache results during development?"

Yes — strongly encouraged. Pickle the results once after the first
successful run, then load from cache during iteration on the
scoring/comparison steps. Same pattern as the W2 caching note.

---

## Writeup + submission

### "How long should the writeup be?"

1 page in markdown. About 300-500 words. Four short sections, one
per reflection question. Tight prose. Don't pad.

### "Should the writeup include the comparison table?"

The table itself lives in the notebook (or `mp1_comparison.md`).
The writeup *references* the table — *"the structured strategy
won on accuracy at 0.9, but cost 3× more than zero-shot"*. Don't
duplicate the full table inside the writeup.

### "Do I need an ADR?"

Optional. If you've already committed to a prompting strategy for
your capstone, write a short ADR (`docs/adr/0002-prompting-strategy.md`)
citing your MP1 findings. Strong submissions often include this.

### "I haven't finished — can I submit partial?"

Yes. Partial submission is better than no submission. The rubric
allows for partial credit on Completion. Submit what you have, note
what's missing, and explain why in the writeup.

### "Can I work with a partner?"

MP1 is individual. You can discuss approach and debug together,
but each learner submits their own writeup and notebook. We expect
to see different prompts, different observations, different
recommendations.

---

## Common errors

### `RuntimeError: This event loop is already running`

You're inside Jupyter. Use `await run_all()` directly, not
`asyncio.run(run_all())`.

### `json.JSONDecodeError: Expecting value: line 1 column 1`

Your model returned text that isn't pure JSON — probably wrapped
in ```json ... ``` fences, or has a preamble like *"Here's the
JSON:"*. Strip the fences and preamble before parsing.

For the chain-of-thought strategy, this is *expected* — the model
will reason in prose before the JSON. Your parser needs to find
the JSON object inside the response.

### `openai.RateLimitError`

Rare for 40 calls. If it happens, add a tiny `await asyncio.sleep(0.1)`
between batches, or split the 40 calls into two batches of 20.

### My structured prompt returns weird hallucinations

Common cause: the JSON schema in the prompt has fields the model
"wants to fill" even when the data isn't there. j10 is the
canonical trap — *"PhD preferred but not required"* gets mis-parsed
as 4 or 5 years.

Mitigation: add an explicit instruction. *"If a field is not stated
in the snippet, return null for that field — do not infer or
guess."* This is itself a finding worth noting in the writeup.

---

*If your question isn't here, post it in the cohort channel. The
instructor will add it to this file.*
