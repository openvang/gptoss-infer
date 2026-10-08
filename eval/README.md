# eval/: the PR evaluation mechanism

This directory is maintainer-owned. A PR that touches it is not scored.

| File | Runs where | What |
|---|---|---|
| `policy.py` | everywhere | every threshold: golden gates, the 24 GiB VRAM budget, tier bands, 99 % paired intervals |
| `bot.py` | trusted host with `gh` | runs rounds: sorts open PRs into lanes, checks RTX 5090 proof, ships each waiting PR (merged onto `main`) to the GPU box as a git bundle, runs `run_eval.py` there, posts and labels the verdict, merges the round's largest verified speedup, closes what can't be scored, enforces the open-PR limit and the stale close |
| `run_eval.py` | GPU box host | exports both commits, builds and measures each in fresh containers, judges, writes `verdict.json` |
| `driver/driver.py` | inside the eval container | drives one build's C API, writes raw results; judges nothing |
| `image/Dockerfile` | GPU box | the eval image: a digest-pinned CUDA 13.0 devel base, plus cmake, ninja, python3 and numpy |
| `tests/test_policy.py` | anywhere | policy unit tests, including an A/A false-tier rate check |
| `tests/test_bot.py` | anywhere | bot rounds against an in-memory GitHub and GPU box |

## Trust model

- **Credentials.** GitHub credentials stay on the bot host. The GPU box gets commits as bundles and returns only
  the verdict.
- **Containers.** Candidate code runs only inside containers started with `--network none`. The weights, golden
  data and harness are mounted read-only; each run gets its own output directory; nothing persists between runs.
- **The harness comes from `main`.** It is extracted from `main` for every evaluation. The golden file is a
  release artifact, accepted only if its sha256 matches the committed `reference/goldens/golden_v1.json`.
- **Judging is outside the container.** Correctness is computed from raw dumps; memory from `nvidia-smi` sampled
  by the driver; timing integrity by comparing the timed runs' log-probs with `main`'s.
- **Merges** are pinned to the evaluated head (`--match-head-commit`), use no admin bypass, and happen only
  while `main` is unchanged since the evaluation.

## GPU box setup

The box used so far is a Lium RTX 5090. Docker volumes must live outside its gocryptfs `/root`.

```bash
mkdir -p /data/models /data/gptoss-eval/{runs,goldens}
python reference/scripts/fetch_weights.py --out /data/models/gpt-oss-20b         # verified against the lock
cp golden_v1.safetensors /data/gptoss-eval/goldens/                               # release artifact
docker build -t gptoss-eval:1 eval/image
```

## Running

```bash
# one evaluation by hand, on the box
python eval/run_eval.py --repo /data/gptoss-eval/repo --base <sha> --cand <sha> --models /data/models \
    --goldens /data/gptoss-eval/goldens --work /data/gptoss-eval/runs

# the bot, on the trusted host
python eval/bot.py --repo openvang/gptoss-infer --box root@<host> --port <port> --key ~/.ssh/<key>
```

One evaluation takes about 12 minutes:
- two builds;
- two golden runs over both paths;
- 5 × 2 timed runs.
