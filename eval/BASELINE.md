# Baseline: `main` @ 8955f73 (2026-10-13)

26 questions on frozen pages (`eval/fixtures/`: python_tutorial and fastapi_tutorial crawled once, 12 pages each; shop and shop_ar handwritten).
Old flow: `Index.top(q, 6)` + `stream_answer`, model `google/gemini-3.8-flash` (low thinking). Tokens from the OpenRouter usage field.

| Metric | Baseline |
|---|---|
| Retrieval hit@4 (22 answerable) | 18/22 (82%): single 14/14, follow-up 3/5, FR/AR 1/3 |
| No-match in retrieval (4 off-site questions) | 0/4 (always returns passages) |
| Answer correct (all 26) | 21/26 (81%) |
| Answerable correct | 17/22 (77%); 5 false "couldn't find" |
| Honest no-answer (off-site) | 4/4 |
| By group | single 18/18, follow-up 2/5, FR/AR 1/3 |
| Model calls | 26 (4 on off-site questions) |
| Input / output tokens | 30,356 / 2,442 (avg 1,168 in, 94 out per call) |
| Follow-up input tokens (avg) | 1,142 |
| Cost | $0.032 for 26 questions |
| Time to first token | median 1.46 s, max 3.48 s |
| Total answer time | median 2.12 s |

Main gaps: follow-ups ("how much does it cost?") search without the previous question; FR/AR questions on English pages find nothing; off-site questions still cost a model call.

Re-run: `python eval/run.py --mode full --tag NAME [--code-dir PATH]` (key in `OPENROUTER_API_KEY`), or `--mode retrieval` with no key. `--rescore --tag NAME` re-applies scoring to a saved run.
