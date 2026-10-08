# Contributing

**Scored contributions are not open yet.** The runtime and the scoring bot don't exist. This file states the
rules they will follow, so that the instruments are built to match.

## How a pull request will be scored

1. **Opt in with evidence.** Tick "tested on RTX 5090" and paste before/after numbers. The bot re-measures
   everything; your numbers only decide whether it spends GPU time.
2. **Isolated measurement.** `main` and your PR are built in fresh, separate containers on the same card. Each
   container has no credentials and no network, and the weights are mounted read-only and hash-checked. A driver
   outside your build sends the workload through the server API and times it.
3. **Correctness first, on every scored axis.**
   - The golden gate: top-1 agreement and KL against `reference/` goldens.
   - Token-exact output against `main` on that axis's own workload: packed vs single-row, cache hit vs miss, long
     context.
   - Harmony and tool-call tests.

   A faster path that changes outputs scores zero.
4. **Statistics, not single runs.**
   - `main` and the PR run interleaved, at least 5 pairs per axis.
   - A gain counts only when its confidence interval clears the noise floor, which is measured nightly from
     `main`-vs-`main` runs.
   - Multiple axes are corrected for.
   - L and XL candidates get a confirmation run.
5. **A persistent frontier.** Each axis keeps its best-ever measurement, and tiers are earned only above it.
   Restoring performance that `main` lost is a bug fix: it is welcome, but it is not paid as a speedup.
6. **A human merges.** The bot labels a PR and recommends merge-ready; a maintainer reads the diff and merges.
   The bot cannot bypass branch protection.
7. **Pay follows the merged commit.** The tier comes from the signed verdict for the exact SHA that was merged.

## Axes (planned)

- **At launch:** single-stream decode at 4k and 32k context, with a floor at 128.
- **With batched prefill:** prefill at 4k and 32k.
- **With the serving engine:** output tokens per second for chat 1024/256 at 4, 16 and 64 concurrent requests,
  with a time-to-first-token guard.
- **Research axis:** speculative decoding with a drafter, required to be byte-identical to plain decode under
  greedy sampling.
- **Guards throughout:** the golden gate, harmony and tool-call tests, 128k decode, and packed-vs-single-row
  equivalence.

## Not allowed

- **Changing what an axis measures.** For example, swapping the drafter on a speculative axis, or tuning to the
  benchmark prompt.
- **Copying another PR's diff,** and running several accounts. Both are detected and blocked.
- **Touching maintainer-owned paths in a scored PR:** `reference/`, `eval/`, `docker/manifest.yaml`, `.github/`.
  Propose changes to those in an issue.
