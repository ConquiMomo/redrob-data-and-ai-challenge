# Redrob Hackathon — Candidate Ranking Pipeline

A two-stage candidate ranking system for the Redrob India "Runs Data \& AI"
hackathon. Ranks \~100,000 candidates against a Search/Ranking/Retrieval ML
Engineer job description and produces a top-100 ranked `submission.csv`.

## Quick start (reproduce the submission)

```bash
pip install -r requirements.txt
python precompute.py    # slow stage — see "Two-stage design" below
python rank.py           # fast stage — produces submission.csv, timed against the 5-min budget
```

Both commands assume `candidates.jsonl` is present in the repository root
(or wherever `CANDIDATES\_FILE` in `precompute.py` is pointed). The output is
written to `submission.csv` in the repository root.

**Single command** (per spec Section 10.3):

```bash
python precompute.py \&\& python rank.py
```

## Two-stage design

The hackathon spec (Section 10.3) explicitly allows pre-computation to
exceed the 5-minute ranking budget, as long as it's documented and run
separately from the timed ranking step. We split accordingly:

|Stage|File|What it does|Time budget|
|-|-|-|-|
|**Precompute**|`precompute.py`|Loads all 100K candidates, applies hard filters, builds per-candidate text, computes bi-encoder embeddings, structured/behavioral/consistency scores. Writes `precomputed.json`.|No hard limit (observed: \~15-25 min on a 16GB CPU machine)|
|**Rank**|`rank.py`|Reads `precomputed.json`, combines signals with current weights, reranks the top \~500 with a cross-encoder (2 weighted passes — see below), writes `submission.csv`.|**Must be ≤5 min.** Observed: \~150-200s on the same machine.|

This split means weight/scoring tuning during development only requires
re-running the fast `rank.py` stage, not the slow embedding stage.

## Compute environment this was tested on

See `submission\_metadata.yaml` for the exact platform/OS/Python version
used for the timing numbers above. No GPU is used at any point; both
models (`BAAI/bge-small-en-v1.5` for the bi-encoder,
`cross-encoder/ms-marco-MiniLM-L12-v2` for cross-encoder reranking) run on
CPU. `rank.py` has no network calls once the models are present in the
local HuggingFace cache (they are downloaded automatically by
`precompute.py`'s first run, which does require network access once).

## Repository contents

```
precompute.py                 # Stage 1: slow, produces precomputed.json
rank.py                        # Stage 2: fast, produces submission.csv
validate\_submission.py         # Organizer-provided validator, included for self-check
jd\_distilled.txt               # Distilled JD text used for the primary semantic-similarity reference
ideal\_candidate.txt            # "Ideal candidate" reference text (secondary semantic signal)
negative\_jd.txt                # "Anti-pattern" reference text (penalizes wrong-specialization / IT-services-only / keyword-stuffed matches)
ce\_signal\_1.txt                # Cross-encoder reference: production systems ownership (weight 0.60)
ce\_signal\_2.txt                # Cross-encoder reference: offline/online evaluation rigor (weight 0.40)
requirements.txt
submission\_metadata.yaml
README.md
```

`candidates.jsonl` (the dataset) is not committed to this repo — see
`.gitignore`. Place it in the repo root before running `precompute.py`.

## Design notes (for Stage 4/5 review)

A few non-obvious decisions, documented here since they came up repeatedly
during development and are worth being able to explain:

* **Location scoring is multiplicative, not additive.** An earlier version
scored location as one of several additive structured-score components
(weight \~0.07), which made it mathematically too weak to meaningfully
affect ranking even for candidates the JD clearly deprioritizes (e.g.,
outside India with no visa sponsorship). It was moved to a direct
multiplier on the final base score (`compute\_location\_mult` in
`rank.py`), the same architectural pattern already used for the
consistency score, disqualifier multiplier, and availability multiplier.
* **Behavioral signals are a down-weighting multiplier, not a scoring
pillar.** The JD's `redrob\_signals` documentation states explicitly that
these signals should be used "as a multiplier or modifier on top of
skill-match scoring," not as an independent positive-scoring axis. A
candidate with normal activity/responsiveness gets no bonus; only
genuinely stale or unresponsive profiles (>90 days inactive, <15%
recruiter response rate) are down-weighted.
* **Skill corroboration is checked at the category level, not the
exact-tool-name level.** An early version of this check required a
candidate's career-history text to contain the literal skill name (e.g.
"Qdrant") to count as corroborated. This produced false positives for
genuinely strong candidates who used one vector DB but also *tagged* a
different one as a skill, or who described relevant work without
narrating every specific tool name. The check now asks whether the
candidate's broad specialty (retrieval/vector-DB, general ML, or
CV/speech) is represented *conceptually* anywhere in their career
history, using word-boundary-aware matching (to avoid short-token
substring false positives, e.g. "rag" inside "storage").
* **The dataset reuses a small pool of job-description templates across
many candidates** (confirmed by direct inspection of the full
`candidates.jsonl`). This is treated as expected dataset texture, not a
fraud signal in itself — penalizing template reuse directly would punish
candidates for which template they happened to be assigned, which is
exactly the kind of shallow-proxy judgment the JD's "read between the
lines" section warns against. Reasoning-text phrasing is varied
deterministically (hashed on `candidate\_id`) so that two candidates
sharing the same underlying template don't produce visibly identical
reasoning strings on manual review.

## Honeypot / consistency checks

* `profile\_consistency\_score`: proficiency-vs-duration mismatches,
endorsement-count uniformity (low coefficient of variation at high mean
endorsement counts), within-candidate duplicate job descriptions.
* `honeypot\_date\_logic\_score`: skill duration exceeding total career span,
multiple simultaneous "current" jobs, overlapping job date ranges.
* `yoe\_career\_mismatch\_check`: stated `years\_of\_experience` vs. what
`career\_history` durations actually sum to — catches a discrepancy
neither the bi-encoder nor cross-encoder can detect on their own, since
neither ever sees the `years\_of\_experience` field directly.

Note: the dataset's `career\_history` schema has no company-founding-year
field, so the "8 years of experience at a company founded 3 years ago"
honeypot pattern described in the spec's example cannot be checked against
this schema.

## Validating your own output

```bash
python validate\_submission.py submission.csv
```

This is the organizers' own validator script (included here for
convenience), checking row count, score ordering, and tie-break rules
(score descending, then `candidate\_id` ascending).

