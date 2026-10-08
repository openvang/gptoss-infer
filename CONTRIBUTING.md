# Contributing

Make gpt-oss-20b faster on the RTX 5090 without changing what it computes. Every pull request is measured
automatically on a dedicated RTX 5090.
- **Faster, with accuracy kept:** the PR is labelled with a tier and merged.
- **Otherwise:** it is closed with the measurements.

## How a PR is evaluated

The eval bot picks up each new PR head (drafts and PRs labelled `hold` are skipped). It merges the PR onto the
current `main` and evaluates that result against `main` (`eval/run_eval.py`):

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
| `eval:XS` … `eval:XL` | Merged (squash) at the evaluated commit. If `main` moved meanwhile, the PR is re-evaluated first. |
| `eval:none`, `eval:REJECT` | Closed, with the measurements in a comment. Push a fix and reopen to be evaluated again. |
| Conflict with `main` | Comment asking for a rebase. Stays open. |
| Touches maintainer-owned paths | `eval:skipped`. Stays open for a maintainer. Not scored. |

PRs from org members and collaborators are labelled but never closed by the bot.

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
./build/gptoss-bench /path/to/gpt-oss-20b
```
