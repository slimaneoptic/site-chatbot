# Wave 2 vs baseline (2026-10-13)

Same 26 questions and frozen pages as `BASELINE.md`, model `google/gemini-3.8-flash` (low thinking).
New flow: `Index.search` (4 passages max, relevance cutoff, previous question, translation) + `answer.ask` (trimmed passages, short history, follow-ups).
Model latency drifts during the day, so `main` was run again right before wave 2 (paired runs).

| Metric | Baseline 12:39 (`main`) | Baseline re-run (`main`, paired) | **Wave 2** (paired) | Wave 2, earlier run |
|---|---|---|---|---|
| Answer correct (26) | 21/26 (81%) | 21/26 | **26/26 (100%)** | 25/26 |
| By group: single / follow-up / FR+AR | 18/18, 2/5, 1/3 | 18/18, 2/5, 1/3 | **18/18, 5/5, 3/3** | 18/18, 5/5, 2/3 |
| Retrieval hit@4 (22 answerable) | 18/22 (82%) | 18/22 | **22/22 (100%)** | 22/22 |
| No-match in retrieval (4 off-site) | 0/4 | 0/4 | **3/4** | 3/4 |
| Model calls (incl. translations) | 26 (4 off-site) | 26 (4 off-site) | **25** (23 answers + 2 translations; 1 off-site) | 25 |
| Input tokens | 30,356 | 30,356 | **13,801 (-55%)** | 13,803 |
| Output tokens | 2,442 | 2,460 | **2,040 (-16%)** | 2,216 |
| Follow-up input tokens (avg) | 1,142 | 1,142 | **634** | 634 |
| Cost | $0.0319 | $0.0320 | **$0.0179 (-44%)** | $0.0186 |
| First token, median | 1.46 s | 1.63 s | **1.56 s** | 1.60 s |
| First token, max | 3.48 s | 3.31 s | 10.71 s (provider stall, see below) | 12.9 s |
| Total answer time, median | 2.12 s | 2.36 s | **2.21 s** | 2.33 s |

Files: `results/wave2.json` (paired run), `results/baseline-rerun.json`, `results/baseline.json`.

## Eval gate
- Better on every quality, token and cost metric.
- First-token median: 1.56 s vs 1.63 s for `main` run at the same time (1.46 s in the 12:39 baseline was a faster API moment).
- First-token max: one item (`fa-pathparam`) stalled 10-13 s at the provider in two full runs. Re-run alone it took 1.17 s and 2.00 s; search took 0.4 ms. Not caused by the code; worth watching in the app.
- Translation latency is now included: the translation call takes 0.7-1.4 s, so the two translated questions see the first token at 1.9-3.1 s.

## Fixes made during the gate
- `search.trim_passage` split hard-wrapped lines as sentences and cut "Control-D ... quit()" out of `py-exit` (wrong answer). It now joins wrapped lines.
- The answer prompt asked for ~120 words; output tokens went up 57%. Now "two to four sentences (under 80 words)" with short follow-ups: output below baseline.
- `eval/run.py` now counts translation calls in calls, tokens, cost and adds search time to the latency (before, translation usage was collected but dropped).

## Remaining gap
- `shop-giftcard` (off-site) still matches one passage, so it costs 1 call; the model answers honestly that gift cards are not mentioned.
