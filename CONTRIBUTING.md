# Contributing

Make gpt-oss-20b faster on the RTX 5090 without changing what it computes. Pull requests that change the
runtime are measured automatically, in rounds, on a dedicated RTX 5090.
- **Faster, with accuracy kept:** the PR earns a tier label. Each round's largest gain is merged.
- **Otherwise:** it is closed with the measurements.

## Before the GPU: lanes and proof

The bot sorts each PR by the files it changes:

| The PR changes | What happens |
|---|---|
| only `runtime/` and `CMakeLists.txt` | scored: evaluated once it has RTX 5090 proof |
| runtime code and other files | not evaluated until you split it: a scored PR plus a separate one |
| no runtime code (docs, tools, scripts) | not scored; a maintainer reviews it |
| any maintainer-owned path | `eval:skipped`; a maintainer reviews it |

A scored PR needs **RTX 5090 proof** in its description. The PR template has the section:
- Tick `- [x] Tested on RTX 5090`.
- Fill the before/after table with `gptoss-bench` numbers from your own RTX 5090. Keep the row labels: the bot
  reads them. At least one column must show after > before.

| Proof | What happens |
|---|---|
| box not ticked | the PR is closed; tick it, fill the table and reopen |
| box ticked, no column faster | `needs-benchmark`: not evaluated; the description is re-read every round |

Ticking the box without running on an RTX 5090 is grounds for blocking the account.

## How a PR is evaluated

Each round, the bot merges every waiting PR onto the same `main` and evaluates that result against `main`
(`eval/run_eval.py`):

1. **Isolation.** Both commits are built and run in fresh containers with no network, read-only weights and no
   credentials. The harness and the golden data always come from `main`, never from your PR.
2. **Correctness.** The golden corpus (`reference/goldens/golden_v1`, 9,278 positions) is run through both
   `gptoss_step` (decode) and `gptoss_prefill`. Each path must pass every check in `eval/policy.py`.

   | Check (each path) | Requirement |
   |---|---|
   | Generated-token top-1 agreement | ≥ 0.993 |
   | Generated-token mean KL | ≤ 1e-3 |
   | Generated-token p99 KL | ≤ 0.012 |
   | All-position mean KL | ≤ 0.019 |
   | Generated-token KL against `main`'s | no more than 25 % worse |

3. **Memory.** The engine process may peak at **24 GiB** of the 5090's 32 GiB. This is measured from outside the
   engine. The rest of the card is reserved for the speech-to-text and text-to-speech models planned for the
   same device.
4. **Speed.** Five interleaved `main`/PR pairs on real tokens. The timed runs must compute the same log-probs as
   `main`'s. The axes:

   | Axis | What it measures |
   |---|---|
   | `decode@128` | tok/s decoding 128 tokens after a 128-token prompt |
   | `decode@4k` | the same, after a 4,096-token prompt |
   | `prefill@4k` | tok/s ingesting the 4,096-token prompt through `gptoss_prefill` |

5. **Verdict.** For each axis, the bot computes the 99 % confidence interval of the PR/main ratio. The PR earns
   the tier of its best axis, judged by the interval's low end:

   | Tier | Speedup |
   |---|---|
   | `eval:XS` | ≥ 2 % |
   | `eval:S` | ≥ 3.5 % |
   | `eval:M` | ≥ 6 % |
   | `eval:L` | ≥ 10 % |
   | `eval:XL` | ≥ 18 % |

   The PR gets `eval:REJECT` if any of these happens:
   - an axis regresses by more than 2 % (with confidence);
   - a correctness or memory check fails;
   - the build breaks.

   Otherwise it gets `eval:none`.

## What happens next

| Verdict | Action |
|---|---|
| a tier, and the round's largest gain | `merge-first`: squash-merged at the evaluated commit |
| a tier, but another PR gained more | `re-evaluate`: measured again on the new `main`, so only your gain on top counts |
| `eval:none`, `eval:REJECT` | closed with the measurements; push a new commit and reopen to be evaluated again |
| does not merge cleanly onto `main` | `needs-rebase`: rebase and push |

The round's largest gain is the highest low end of a 99 % interval on any axis. If `main` moves during a round,
nothing merges and every verified PR is measured again on the new `main`.

While your PR waits on the bot, one label shows where it is: `status:queued`, `status:node-starting` (an RTX
5090 node is being rented and set up) or `status:evaluating`. The verdict replaces it.

Limits:
- **Open PRs:** at most 5 per contributor; the newest beyond that are closed.
- **Inactivity:** a PR waiting on you (`needs-benchmark`, `needs-rebase` or a split request) is closed 2 days
  after the bot's last comment if nothing happens. Push a commit and reopen to continue.
- **`hold`:** a maintainer-only label. The bot does not evaluate, merge or close the PR. Drafts are not
  evaluated.
- **Org members and collaborators** skip the proof, the limits and every close.

## Where to work

- `runtime/src/` (kernels, engine) and `runtime/include/gptoss/gptoss.h`. Keep the C API compatible: the harness
  drives your build through it.
- `gptoss_prefill` currently runs one decode step per token. A real batched prefill must produce the same
  distributions, and it will move `prefill@4k`.
- See `runtime/README.md` for the M1 profile (where the time goes).

## What you can't change in a scored PR

Maintainer-owned paths: `eval/`, `reference/`, `docker/manifest.yaml`, `.github/`, `bench/baselines/`. They
define how PRs are measured. Propose changes to them in an issue.

## Not allowed

- **Changing what is measured:** special-casing the golden or bench tokens, or detecting the harness.
- **Changing the model's arithmetic** beyond the correctness bounds above.
- **Copying another PR,** or using several accounts.

Each is grounds for closing PRs and blocking the account.

## Before you open a PR

```bash
cmake -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a && cmake --build build -j
GPTOSS_LIB=build/libgptoss.so PYTHONPATH=reference python -m pytest -q runtime/tests
PYTHONPATH=reference python runtime/tools/score_golden.py --lib build/libgptoss.so \
    --model-dir /path/to/gpt-oss-20b --golden reference/goldens/golden_v1 --out /tmp/cand.safetensors
./build/gptoss-bench /path/to/gpt-oss-20b --depths 128,4096    # on main, then on your branch
```

For the PR template's table, copy each `median_tok_s`: `tg128` at depth 128 (decode@128), `tg128` at
depth 4096 (decode@4k) and `pp4096` (prefill@4k).
