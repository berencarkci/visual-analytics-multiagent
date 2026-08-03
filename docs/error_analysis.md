# Error analysis

Every failure recorded here was observed in a real run, not imagined as a risk.
Each entry names what happened, why, what was done about it, and — where it
matters — what the fix cost. Cases are grouped by root cause rather than by
symptom, because the same underlying mistake surfaced in several places and the
grouping is the finding.

Two sections at the end are deliberately *not* error analysis: the DPO result is
a negative experimental finding, and the scope limits are boundaries of the
design rather than defects in it.

---

## A. Silent data-type failures

A wrong answer that looks well-formed is worse than a crash. Every case in this
group produced a confident sentence over a number that meant nothing.

**A1 — Numeric coercion over a categorical column.** A distribution question on
a text column reached `_compute_stats`, which called `pd.to_numeric(...,
errors="coerce")` and described whatever survived. A column of category labels
containing a handful of numeric-looking values produced "median 6.0" — a median
over the 3% of rows that happened to parse, presented as the median of the
column. Fixed by measuring the coercion failure rate: above 20% non-numeric the
statistics switch to per-category counts and record why.

**A2 — Aggregated output described as a raw distribution.** After a
`groupby` + `agg`, the prepared table holds one aggregate per group, not the
original values. `distribution_stats` computed a median over those aggregates
while still naming the source column, yielding "distribution of age, median
100" — a median of counts labelled as a median of ages. Fixed by switching the
insight focus to group statistics whenever `groupby` and `agg` are both present.

Worth noting: the guardrail that later caught this class of error was itself
invalidated by a subsequent training round. Once the model was taught to emit
`groupby=category, agg=count` for categorical distributions, the condition that
had been detecting the mistake stopped firing — the mistake no longer occurred.
A guardrail's trigger rate is not a stable measurement; it depends on the model
it sits behind.

**A3 — `.sum()` on a text column.** pandas concatenates strings under `.sum()`,
so an aggregation targeting a text column returned
`"ConsumerConsumerCorporate…"` as a bar value while every structural check
passed: the column existed, the chart type was allowed, no number was invented.
Found by the NLV Corpus probe, not by the benchmark. Fixed by downgrading
`sum`/`mean` on a non-numeric target to `count` and recording the substitution.

**A4 — Pearson correlation over a categorical column.** "Is there a correlation
between gender and age" coerced `gender` to NaN, dropped every row, and produced
`r=nan over 0 rows`. The insight template then dressed NaN as "weak and
negative", since the adjectives were derived without checking for finiteness.
The evaluation agent did catch it (`stats_health: NaN r or n=0, insight cannot be
trusted`), so the user was warned — but the warning arrived after a nonsense
sentence had already been written. Fixed at the source: fewer than three usable
rows now yields a `correlation_invalid` result that states a correlation needs
two numeric columns.

**A5 — `quantile` on a boolean dtype.** Raised rather than returning a value.
Fixed by casting boolean to integer before the quantile call. Trivial, but it
only appeared once an uploaded table contained a boolean column — none of the
three sample datasets did.

---

## B. Semantic drift under a passing metric

This is the central failure mode of the project and the one worth the most space
in the report. In every case below the system did not crash and did not invent a
number. It answered a neighbouring, easier question and every mechanical check
approved the result.

**B1 — A filter that can never match.** Asked how much energy appliances use at
night, the model wrote `hour_of_day(date) > 17 and hour_of_day(date) < 7`. No
hour satisfies both; the transform returned zero rows and the chain stopped.
The model had learned the mechanism (`hour_of_day` in a filter) and failed only
on the boolean operator, because a night interval wraps past midnight and needs
a disjunction.

**B2 — The repair attempt made the failure quieter instead of fixing it.** A
retry was added: on an empty result, feed the empty filter back and ask for a
correction. The chain stop disappeared. But the model did not repair the filter
— it deleted it, and answered with a 24-hour profile. The metric improved
(`chain_stopped` went from 1 to 0) while the answer got worse: the question asked
about night and the reply described the whole day. Strengthening the hint
("keep the filter, fix its logic") did not change the behaviour.

**B3 — The correct fix was deterministic, not learned.** `col > a and col < b`
with `a >= b` has no solution, so a disjunction is the only reading under which
the filter means anything. That is a logical certainty, not an inference about
intent, and it belongs in code. A plan guardrail now rewrites the conjunction,
records the rewrite in the trace, and runs *before* the empty-result retry so
the model never gets the chance to drop the filter. The night question returns
13 rows and the guardrail note is visible to the user.

The sequence — train, observe the metric improve while the answer degrades,
then solve it with a rule — is the clearest engineering lesson in the project.

**B4 — Nearest-column substitution on unseen schemas.** The cross-domain probe
asked for three quantities that no column held. In all three the system
substituted the closest available column instead of deriving or declining:

| asked | expected | produced |
|---|---|---|
| how many pupils needed a resit | count over a boolean | grouped by the pupil identifier, 400 rows |
| what fraction of the bill is parts | `ratio(parts_cost, total_billed)` | a composition of parts cost, no ratio |
| average labour cost per hour | no such column | mean of `labour_hours` |

**B5 — Ratio direction inverted.** Unit price is revenue divided by quantity;
the model produced `ratio(Sales_Quantity, Total_Revenue)` — the reciprocal, and
a meaningless 0.02 presented as a unit price. Direction is a semantic decision:
which quantity is the numerator depends on what the question means, and no
deterministic rule recovers it. Rewriting the question as an explicit division
produced the correct plan, so the mechanism works when the phrasing pins it down.

**B6 — Ratio not triggered at all.** "What is total revenue divided by sales
quantity for each category" produced `mean(Total_Revenue)`, with the division
absent entirely. Root cause in section E.

**The common shape.** A question names a derived or conditional quantity; no
column matches it; the system answers with the nearest expressible query. The
groundedness verifier cannot catch any of this, because it checks whether the
numbers in the sentence exist in the computed statistics — it has no
representation of the question. Every number in every case above was real. This
belongs verbatim in the limitations section: *mechanical checks verify that an
answer is internally consistent, not that it answers the question asked.*

---

## C. Validator, engine and prompt out of step

Three components independently held a list of what the system supports, and they
drifted.

**C1 — A mechanism defined in the validator but absent from the engine.**
`threshold_flag` was on the validator's whitelist of derived groupings while
`_resolve_grouping` had no branch for it. A plan using it passed validation, then
the engine skipped the grouping with "Unknown derived grouping" and answered
without it. Found by an automated cross-check of the three lists, which also
revealed that the engine supported nine grouping expressions while the prompt
documented three.

**C2 — Knowledge in the weights that the prompt never stated.** Of those nine,
`hour_of_day`, `quarter` and `weekend_flag` appeared nowhere in the prompt, and
the model used them correctly anyway — `hour_of_day` from five training examples.
This is direct evidence for the project's central claim that format knowledge
moved from the prompt into the weights. It is also a maintenance hazard: a
mechanism present in neither the prompt nor the training data is dead code that
no test would notice.

**C3 — A category value where a column was expected.** The model wrote
`series="Technology"` — a value, not a column. The generic token scan skipped it
because the scan only collected lowercase tokens, and the failure surfaced as a
`KeyError` in the engine. `series` is always a column, never a literal, so it is
now validated strictly and the rejection message tells the model to use a filter
instead. The equivalent hole exists for `groupby` and was never observed; the
underlying weakness is that a validation layer distinguished identifiers from
literals by capitalisation.

**C4 — A date function applied to a categorical column.** `month(region)`, which
raised inside the datetime parser. Caught as a skipped grouping rather than a
crash, but the plan then ran without its time axis.

**C5 — A half-applied relaxation.** Case-insensitive column matching was added
to the validator so `technology` would be accepted for `Technology`. The engine
was left case-sensitive, so an accepted plan then threw `KeyError` deeper in the
pipeline — strictly worse than a clean rejection with feedback, because the error
arrived without usable guidance. Reverted. A validation rule and the execution it
guards have to agree; relaxing one alone converts a clear failure into an obscure
one.

---

## D. Guardrail calibration

Guardrails encode rules about readable output. Two were keyed on the wrong
quantity.

**D1 — Overplotting measured as an absolute count.** The scatter-to-box rule
fired when the x column had twelve or fewer distinct values, on the reasoning
that many points stack on each value. On a 9,994-row table with twelve discount
levels this is correct — roughly 830 points per value. On an 8-row uploaded table
with eight distinct values it produced eight boxes of one point each, with no
quartiles, strictly worse than the scatter it replaced. The real signal is the
ratio: the rule now requires at least four observations per distinct value, which
preserves the original case and drops the degenerate one.

**D2 — An identifier as a colour dimension.** A plan set `series=order_id`,
which would have expanded the grouping to thousands of rows and a legend with one
entry per order. A colour dimension is only readable with a handful of values, so
a plain column with more than twelve distinct values is now dropped from
`series` with a note. Derived series are exempt: `weekend_flag(date)` has two
values regardless of the source column's cardinality.

---

## E. Training-data design defects

These are mistakes in the synthetic data, not in the model. They are the most
transferable lessons in the project because they are consequences of decisions
that were made deliberately.

**E1 — A phrase locked to one meaning.** "Divided by" was used in the `series`
bank, in questions of the form "average profit of each ship mode divided by
segment", where segment is a second grouping dimension. The model learned the
phrase as dimension-splitting rather than division, and when a later question
asked for an actual division it did not reach for `ratio` at all. Over-
representing an expression in a single pattern narrows its meaning. The phrase
has been removed from that bank.

**E2 — A memorised pair instead of a learned abstraction.** All 24 `ratio`
examples used two column pairs, both in the retail dataset. The mechanism fires
reliably on those exact columns — the capability probe passes both ratio cases —
and not at all on an unseen table with different column names. Coverage was
counted by example volume when the property that mattered was the variety of
column pairings.

**E3 — Recited rationale contradicting the transform.** In four of five
adversarial probes the `reason` field explained a mechanism the `transform` did
not use: "the rows split into two groups rather than being filtered" alongside a
plan containing no threshold, and "two breakdowns are named, so Category is the
breakdown and Unit is the weight" alongside a plan with no series. The banks
varied the rationale text but did not teach the pairing between a rationale and
the transform it describes, so the model learned to produce plausible
explanations independently of what it planned.

**E4 — A schema addition creating an ambiguity elsewhere.** Adding the optional
`series` field made the model list the series column in `target_columns` instead
of the measure. `target_columns[1]` is read as the aggregation target, so the
aggregate landed on a text column and was downgraded to a count — "profit per
ship mode by segment" returned row counts. Caught by a guardrail that strips the
series column from `target_columns`, and addressed properly in training data.

**E5 — Rules in the prompt are not enough, measured three times.** Anomaly
phrasing needed the bank to grow from 6 to 43 examples before the behaviour
appeared. An intent rule about subset modifiers over-triggered until contrast
twins were added. And when `threshold_flag` and `series` were documented in the
prompt with no examples at all, the model reached for them in 12 of 30 probes but
placed them correctly in only a minority — after training on examples the same
groups scored 6/6 and 7/7. Documenting a mechanism teaches its name; examples
teach where it does and does not apply.

**E6 — Example count is not a single threshold.** Simple substitutions — one
column mapped to one derived expression — generalised from four or five
examples: `weekend_flag` with four examples and `day_of_week` with five were
correct in all eight probes that exercised them. Structural behaviours needed
around twenty: a derived measure occupying the y axis (`diff`, `ratio`) did not
generalise at low volume. The requirement scales with the complexity of the
behaviour, not with a fixed number.

**E7 — Over-triggering as the dominant cost of a new mechanism.** When `series`
was introduced, the failure was not that the model refused to use it but that it
used it everywhere: on a share question, on a distribution, on a relationship, on
a question with a filter and one dimension. Contrast twins therefore outnumber
positive examples in the final banks.

---

## F. Measurement and infrastructure traps

Failures in how the system was evaluated, rather than in the system.

**F1 — Prompt drift, with its cost measured.** The architecture guarantees that
training and inference inputs are identical, because the data generator imports
each agent's own prompt builder. Editing the prompts without regenerating the
data and retraining violated that invariant by hand. The result was measurable:
guardrail corrections rose from 4 to 6, one chain stop appeared, and a share
question produced 7,897 categories where four regions were expected — because the
model stopped emitting `groupby`. Retraining on the new prompts closed it
completely. An accidental experiment that quantifies why the invariant exists.

**F2 — Error paths that no test had reached.** Running the untrained base model
exercised failure branches the trained model never entered, exposing a
`KeyError` on a review-feedback lookup and two call sites reading a
non-existent `StepError.message` attribute. The weak model functioned as both
control group and bug finder; the trained model's reliability had left its
recovery code unverified.

**F3 — A signature change with a missed call site.** Adding a return value to
`apply_transform` required updating five call sites. One probe was missed, and
its output read `0/8 capability cases passed` — while every one of the eight
plans was correct. A test harness failure presenting as a total model failure;
the plans had to be read individually to see it.

**F4 — Stale code in a live kernel.** `git pull` updates files on disk but not
modules already imported. Two verification runs tested code that had been
replaced, once producing a result that looked like a failed fix. Any pull now
requires a kernel restart before measurement.

**F5 — A counter that measured the wrong thing.** The dev scan reports
`retried 0` for a question where a retry demonstrably occurred, because the
counter tracks orchestrator-level retries and the retry happened inside the Data
Analyst. The honest reading is "no retry escalated to the orchestrator", not "no
retry was needed".

**F6 — Coverage measured over the wrong text.** Counting mechanism usage with a
grep over whole training lines returned 591 — the file's line count — for five
mechanisms, because those names appear in the system prompt embedded in every
example. Real usage was between 4 and 99. The first measurement suggested
saturation where four of the mechanisms were far below the level the project's
own evidence showed to be necessary.

---

## The DPO result

A DPO adapter was trained on 430 preference pairs (270 synthetic, the rest
harvested from real model disagreements) on top of SFT-v2, and evaluated on the
same frozen splits as every other arm.

Training metrics were excellent: reward accuracy reached 1.0 and the margin
reached 4.6. Held-out performance did not move. On the combined 60-question
benchmark, intent accuracy was 55/60 for both SFT-v2 and DPO-all — down one on
dev, up one on test, which is noise. DPO trained only on the harvested real pairs
scored 53/60, matching the untrained baseline.

The reward accuracy is the diagnosis. Reaching 1.0 means the pairs were trivially
separable: the rejected responses were malformed rather than subtly worse, so the
model learned to prefer well-formed output it was already producing. The
remaining errors were missing capabilities, not preference errors — and a
preference signal cannot supply a capability that is absent. When the probe
failures that motivated the pairs were re-measured after DPO, five of eight were
still wrong.

The correct statement is not that DPO regressed, nor that it failed, but that its
effect was inside the noise floor in both directions. In this task at this scale
the gains came from targeted supervised data; preference optimisation polishes a
behaviour that already exists and cannot build one that does not. This matches
the risk registered before the experiment began, which is why it is reported
rather than dropped.

---

## Scope limits, not defects

These are boundaries of the design. Each has a known fix that was judged out of
scope, and none is a case of the system behaving incorrectly within its design.

**User-defined buckets.** `groupby` accepts one column or one derived
expression, so "treat 11–14 March as one group and 15–21 March as another"
cannot be expressed: `day()` gives one group per day, `week()` uses ISO
boundaries, `bins()` chooses its own edges, and `threshold_flag` takes a numeric
column. The system falls back to a daily time series. The workaround is two
filtered single-value queries. The fix spans the schema, the engine, the renderer
and the training data.

**Two measures on different scales.** "How does total revenue compare against
average satisfaction" needs a dual axis or a scatter of two aggregates; neither
exists. The system answers with one measure and says so in its rationale
("averages need their own scale"), which is the correct behaviour available to
it.

**Grouping beyond two dimensions.** `series` adds one extra dimension. A third
would need faceting.

**Stacked versus grouped bars.** Two dimensions render as grouped bars. Grouped
bars support cross-group comparison better than stacked bars, which is the
common recommendation, so this is a defensible default rather than a gap — but it
is a choice, and a stacked variant is a one-word change with no way for the user
to request it.

**Non-English questions.** All insight and rationale training targets are
English. A Turkish question produces broken Turkish prose. The numbers stayed
correct and the groundedness check still passed, which localises the failure
precisely: mechanical verification is language-independent, fluency is not.