# eval/: the PR evaluation mechanism

This directory is maintainer-owned. A PR that touches it is not scored.

| File | Runs where | What |
|---|---|---|
| `policy.py` | everywhere | every threshold: golden gates, the 24 GiB VRAM budget, tier bands, 99 % paired intervals |
| `bot.py` | trusted host with `gh` | runs rounds: sorts open PRs into lanes, checks RTX 5090 proof, ships each waiting PR (merged onto `main`) to the GPU box as a git bundle, runs `run_eval.py` there, posts and labels the verdict, merges the round's largest verified speedup, closes what can't be scored, enforces the open-PR limit and the stale close |
| `run_eval.py` | GPU box host | exports both commits, builds and measures each in fresh containers, judges, writes `verdict.json` |
| `driver/driver.py` | inside the eval container | drives one build's C API, writes raw results; judges nothing |
| `box/provision.sh` | GPU box | prepares a box: GPU and driver check, Docker GPU access, the judge's Python, the verified weights, the eval image |
| `vast.py` | trusted host | rents and releases RTX 5090 VMs on vast.ai through the `vastai` CLI |
| `image/Dockerfile` | GPU box | the eval image: a digest-pinned CUDA 13.0 devel base, plus cmake, ninja, python3 and numpy |
| `tests/test_policy.py` | anywhere | policy unit tests, including an A/A false-tier rate check |
| `tests/test_bot.py` | anywhere | bot rounds against an in-memory GitHub and GPU box |
| `tests/test_vast.py` | anywhere | the rental lifecycle against a fake `vastai` CLI |

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

## The GPU box

The bot prepares any box itself. Before an evaluation, it runs main's `eval/box/provision.sh` there whenever that
script, the eval image, the weights lock or the pinned requirements have changed, and uploads the golden release
artifact if the box lacks it. All box work is plain shell over SSH (`ssh`, `scp`, `git`, `docker`).

**vast.ai.** With `--vast-env`, the bot rents an RTX 5090 VM when a round has PRs to evaluate, reuses it while work
keeps coming, and destroys it after `--vast-idle-minutes` (7) without work. It must be a VM: vast.ai's container
instances can't run Docker.
- **Offers:** one RTX 5090 on a verified host with reliability ≥ 0.98, download ≥ 300 Mb/s, disk ≥ 150 GB,
  RAM ≥ 32 GB and ≥ 8 CPU cores. The bot takes the best DL-perf per dollar at or under `--vast-max-dph`
  ($1.00/h).
- **Credit guard:** it does not rent below $2 of credit.
- **Image:** `vastai/kvm:ubuntu_cli_22.04-2025-11-21`, with NVIDIA driver 580 (CUDA 13.0), Docker with the NVIDIA
  runtime, git and uv.
- **API key:** read from the env file and passed only to the `vastai` CLI process, never to the box.
- **Provisioning:** a fresh VM takes about 5 minutes.

**A fixed box.** `--box user@host --port N` uses an existing machine with the same software. Docker volumes must
live outside an encrypted `/root` (as on Lium); the bot uses `/data`.

## PR status labels

While a PR waits on the bot, one status label shows where it is. The verdict replaces it.

| Label | Meaning |
|---|---|
| `status:queued` | waiting for the next round (also after a failed attempt, which is retried) |
| `status:node-starting` | an RTX 5090 node is being rented and set up for this round |
| `status:evaluating` | being built and measured now |

## Running as a service

The bot runs as a systemd user service that starts at boot (`loginctl enable-linger`). `gh` uses the maintainer
account's keyring login; with `--gh-user`, the bot switches `gh` back to that account whenever another one is active.
SSH uses only the bot's key. The bot runs from its own clone of `main` and pulls it on every start. Every external
call is time-limited, so a hung command can't stall it. With `--restart-on-update`, it exits after any round where
`main` has changed `eval/`, so systemd restarts it on the new code.

```ini
# ~/.config/systemd/user/gptoss-eval-bot.service
[Unit]
Description=gptoss-infer PR eval bot

[Service]
Environment=APP=%h/.local/share/gptoss-eval-bot/app
UnsetEnvironment=SSH_AUTH_SOCK
ExecStartPre=-/usr/bin/git -C ${APP} pull --ff-only -q
ExecStart=/usr/bin/python3 -u ${APP}/eval/bot.py --repo openvang/gptoss-infer --gh-user <account> --restart-on-update \
    --vast-env <env file with VAST_API_KEY> --vast-cli <vastai> --key <ssh key> \
    --golden %h/.local/share/gptoss-eval-bot/golden_v1.safetensors
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
```

```bash
git clone -q https://github.com/openvang/gptoss-infer ~/.local/share/gptoss-eval-bot/app
cp reference/goldens/golden_v1.safetensors ~/.local/share/gptoss-eval-bot/     # release artifact
systemctl --user daemon-reload && systemctl --user enable --now gptoss-eval-bot
journalctl --user -u gptoss-eval-bot -f                                          # the bot's log
```

`gh` must stay logged in as that account, and its keyring must be unlocked: after a reboot, a login unlocks it. One evaluation by hand, on a provisioned box:

```bash
/data/venv/bin/python eval/run_eval.py --repo /data/gptoss-eval/repo --base <sha> --cand <sha> \
    --models /data/models --goldens /data/gptoss-eval/goldens --work /data/gptoss-eval/runs
```

One evaluation takes about 12 minutes:
- two builds;
- two golden runs over both paths;
- 5 × 2 timed runs.
