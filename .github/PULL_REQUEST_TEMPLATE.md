## What changed

<!-- One optimization per PR keeps the measured gain attributable. A scored PR changes only runtime/ and
     CMakeLists.txt. -->

## RTX 5090 results

- [ ] Tested on RTX 5090

<!-- Run `./build/gptoss-bench <model-dir> --depths 128,4096` on main and on this branch. Copy each median_tok_s:
     tg128 at depth 128, tg128 at depth 4096 and pp4096. Keep the row labels: the bot reads them. The PR is
     evaluated only with the box ticked and a column where after > before. -->

| | decode@128 | decode@4k | prefill@4k |
|---|---:|---:|---:|
| before (main) | | | |
| after (this PR) | | | |

## Checks run locally

- [ ] `runtime/tests` pass
- [ ] `score_golden.py` within the bounds in CONTRIBUTING.md
