#!/usr/bin/env python3
"""PR bot: evaluate every new pull-request head on the GPU box against main, label it with the verdict, merge it
when it is a verified speedup that keeps accuracy, and close it when it is not.

Runs on a trusted host where `gh` is logged in as the maintainer account. The GPU box never receives GitHub
credentials: the bot ships commits to it as a git bundle over SSH, runs main's eval/run_eval.py there, and reads
back only the verdict JSON.

    python eval/bot.py --repo openvang/gptoss-infer --box root@HOST --port PORT --key ~/.ssh/key [--once]

A PR is evaluated as it would land: merged onto the current main. Draft PRs and PRs labelled `hold` are skipped.
A tier is merged only if main has not moved since the evaluation (otherwise it is re-evaluated) and only at the
evaluated head commit. `none` and `REJECT` close the PR, except for org members and collaborators, whose PRs
(docs, refactors, harness work) stay open for a maintainer. PRs that were not measured (maintainer-owned paths,
merge conflicts, infrastructure errors) are never closed by the bot.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy  # noqa: E402

BOX_ROOT = "/data/gptoss-eval"
MARKER = "<!-- gptoss-eval head={head} base={base} label={label} -->"
MARKER_RE = re.compile(r"<!-- gptoss-eval head=(\w+) base=(\w+) label=([\w:-]+) -->")
MEMBERS = {"OWNER", "MEMBER", "COLLABORATOR"}
COLORS = {"XL": "0e8a16", "L": "2cbe4e", "M": "6ad17a", "S": "9fe2a6", "XS": "c9f1cd", "none": "bfbfbf",
          "REJECT": "d73a4a", "skipped": "fbca04"}


def run(cmd, input=None, timeout=None):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, input=input, timeout=timeout).stdout


class Bot:
    def __init__(self, args):
        self.a = args
        self.work = Path(args.workdir).expanduser()
        self.mirror = self.work / "mirror"
        self.ssh = ["ssh", "-i", str(Path(args.key).expanduser()), "-p", str(args.port), "-o", "BatchMode=yes",
                    args.box]
        self.scp = ["scp", "-q", "-i", str(Path(args.key).expanduser()), "-P", str(args.port), "-o", "BatchMode=yes"]

    def gh(self, *args, input=None):
        return run(["gh", *args], input=input)

    def git(self, *args, cwd=None):
        return run(["git", "-C", str(cwd or self.mirror), *args]).strip()

    def setup(self):
        self.login = self.gh("api", "user", "--jq", ".login").strip()
        self.work.mkdir(parents=True, exist_ok=True)
        if not self.mirror.exists():
            run(["gh", "repo", "clone", self.a.repo, str(self.mirror), "--", "-q"])
            # Local merge commits only (never pushed) need an identity.
            self.git("config", "user.name", "gptoss-eval-bot")
            self.git("config", "user.email", "gptoss-eval-bot@localhost")
        existing = {l["name"] for l in json.loads(self.gh("label", "list", "--repo", self.a.repo, "--json", "name",
                                                           "--limit", "200"))}
        for key, name in policy.LABELS.items():
            if name not in existing:
                self.gh("label", "create", name, "--repo", self.a.repo, "--color", COLORS[key],
                        "--description", f"gptoss eval verdict: {key}")
        if "hold" not in existing:
            self.gh("label", "create", "hold", "--repo", self.a.repo, "--color", "d93f0b",
                    "--description", "do not evaluate or merge")

    def main_sha(self):
        return self.gh("api", f"repos/{self.a.repo}/commits/main", "--jq", ".sha").strip()

    def markers(self, number):
        # Only the bot's own comments count: anyone can post a comment that looks like a marker.
        jq = f'.[] | select(.user.login == "{self.login}") | .body'
        bodies = self.gh("api", f"repos/{self.a.repo}/issues/{number}/comments", "--paginate", "--jq", jq)
        return MARKER_RE.findall(bodies)

    def needs_eval(self, pr, main):
        done = [(b, lbl) for h, b, lbl in self.markers(pr["number"]) if h == pr["head"]]
        if not done:
            return True
        base, label = done[-1]
        # A tier that could not merge because main moved is re-measured against the new main.
        return label in policy.TIERS and base != main

    def candidate(self, pr, main):
        """Merge the PR head onto main locally; returns the merged commit, or None on conflict."""
        n = pr["number"]
        self.git("fetch", "-q", "origin", f"+refs/heads/main:refs/remotes/origin/main",
                 f"+refs/pull/{n}/head:refs/eval/pr{n}")
        wt = self.work / f"wt-{n}"
        if wt.exists():
            self.git("worktree", "remove", "--force", str(wt))
        self.git("worktree", "add", "-q", "--detach", str(wt), main)
        try:
            run(["git", "-C", str(wt), "merge", "-q", "--no-ff", "--no-edit", f"refs/eval/pr{n}"])
            return self.git("rev-parse", "HEAD", cwd=wt)
        except subprocess.CalledProcessError:
            return None
        finally:
            self.git("worktree", "remove", "--force", str(wt))

    def evaluate_on_box(self, main, cand):
        self.git("update-ref", "refs/eval/base", main)
        self.git("update-ref", "refs/eval/cand", cand)
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "eval.bundle"
            self.git("bundle", "create", str(bundle), "refs/eval/base", "refs/eval/cand")
            run(self.scp + [str(bundle), f"{self.a.box}:{BOX_ROOT}/incoming.bundle"])
        q = shlex.quote
        remote = (f"set -e; cd {BOX_ROOT}; [ -d repo ] || git init -q repo; "
                  f"git -C repo fetch -q {BOX_ROOT}/incoming.bundle '+refs/eval/*:refs/eval/*'; "
                  f"rm -rf harness && mkdir harness && git -C repo archive {q(main)} | tar -x -C harness; "
                  f"{self.a.box_python} harness/eval/run_eval.py --repo repo --base {q(main)} --cand {q(cand)} "
                  f"--models {self.a.box_models} --goldens {BOX_ROOT}/goldens --work {BOX_ROOT}/runs "
                  f"--pairs {self.a.pairs}")
        proc = subprocess.run(self.ssh + [remote], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=4 * 3600)
        lines = proc.stdout.strip().splitlines()
        if not lines or not lines[-1].endswith("verdict.json"):
            raise RuntimeError(f"eval failed on the box (exit {proc.returncode}): {proc.stderr[-1500:]}")
        with tempfile.TemporaryDirectory() as d:
            run(self.scp + [f"{self.a.box}:{lines[-1]}", f"{d}/verdict.json"])
            return json.loads(Path(d, "verdict.json").read_text())

    def comment(self, pr, verdict, main):
        label = verdict["label"]
        lines = [f"**gptoss eval: `{policy.LABELS.get(label, label)}`**", ""]
        if verdict.get("axes"):
            lines += ["| axis | base | candidate | ratio (99 % CI) | tier |", "|---|---:|---:|---|---|"]
            for axis, s in verdict["axes"].items():
                lines.append(f"| {axis} | {s['base_median']:.1f} | {s['cand_median']:.1f} | {s['ratio']:.3f} "
                             f"({s['low']:.3f}–{s['high']:.3f}) | {s['tier'] or '—'} |")
            lines.append("")
        if verdict.get("golden"):
            for path in ("decode", "prefill"):
                s = verdict["golden"]["cand"][path]
                lines.append(f"- golden ({path}): gen top-1 {s['gen_top1']:.4f}, gen KL {s['gen_kl_mean']:.2e}, "
                             f"p99 {s['gen_kl_p99']:.2e}")
        if verdict.get("vram_peak_mib"):
            lines.append(f"- VRAM peak {verdict['vram_peak_mib']['cand']} MiB (budget {verdict['vram_budget_mib']} MiB)")
        for r in verdict.get("reasons", []):
            lines.append(f"- {r}")
        gpu = verdict.get("gpu", {})
        lines += ["", f"<sub>policy v{verdict['policy_version']} · base `{main[:8]}` · candidate (merged onto base) "
                      f"`{verdict['cand'][:8]}` · {gpu.get('name', '?')} at {gpu.get('power.limit', '?')} · run "
                      f"`{verdict['run']}`</sub>",
                  MARKER.format(head=pr["head"], base=main, label=label)]
        self.gh("pr", "comment", str(pr["number"]), "--repo", self.a.repo, "--body-file", "-", input="\n".join(lines))

    def set_label(self, pr, label):
        stale = [n for n in pr["labels"] if n.startswith("eval:") and n != policy.LABELS[label]]
        args = ["pr", "edit", str(pr["number"]), "--repo", self.a.repo, "--add-label", policy.LABELS[label]]
        for name in stale:
            args += ["--remove-label", name]
        self.gh(*args)

    def open_prs(self):
        jq = ('.[] | {number, head: .head.sha, draft, base: .base.ref, labels: [.labels[].name], '
              'association: .author_association, author: .user.login}')
        out = self.gh("api", f"repos/{self.a.repo}/pulls?state=open&per_page=100", "--paginate", "--jq", jq)
        return sorted((json.loads(line) for line in out.splitlines() if line.strip()), key=lambda p: p["number"])

    def run_once(self):
        for pr in self.open_prs():
            n = pr["number"]
            if pr["draft"] or pr["base"] != "main" or "hold" in pr["labels"]:
                continue
            main = self.main_sha()
            if not self.needs_eval(pr, main):
                continue
            print(f"evaluating #{n} {pr['head'][:8]} onto main {main[:8]}", flush=True)
            cand = self.candidate(pr, main)
            if cand is None:
                self.gh("pr", "comment", str(n), "--repo", self.a.repo, "--body",
                        "gptoss eval: this PR does not merge cleanly onto main; please rebase.\n"
                        + MARKER.format(head=pr["head"], base=main, label="conflict"))
                continue
            try:
                verdict = self.evaluate_on_box(main, cand)
            except Exception as e:                        # infrastructure problem: retry next round
                print(f"  #{n}: {e}", flush=True)
                continue
            label = verdict["label"]
            if label == "error":
                print(f"  #{n}: infra error {verdict.get('reasons')}", flush=True)
                continue
            self.comment(pr, verdict, main)
            self.set_label(pr, label)
            print(f"  #{n}: {label}", flush=True)
            if label in policy.TIERS:
                if self.main_sha() != main:
                    print(f"  #{n}: main moved during evaluation; re-evaluating next round", flush=True)
                    continue
                try:
                    self.gh("pr", "merge", str(n), "--repo", self.a.repo, "--squash", "--match-head-commit", pr["head"])
                    print(f"  #{n}: merged", flush=True)
                except subprocess.CalledProcessError as e:
                    print(f"  #{n}: merge refused: {e.stderr.strip()[:300]}", flush=True)
            elif label in ("none", "REJECT"):
                if pr["association"] in MEMBERS:
                    print(f"  #{n}: {pr['association'].lower()} PR left open", flush=True)
                else:
                    self.gh("pr", "close", str(n), "--repo", self.a.repo)
                    print(f"  #{n}: closed", flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--box", required=True, help="user@host of the GPU box")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--key", required=True)
    ap.add_argument("--box-python", default="/root/work/tvenv/bin/python")
    ap.add_argument("--box-models", default="/data/models", help="directory holding gpt-oss-20b on the box")
    ap.add_argument("--workdir", default="~/.cache/gptoss-eval-bot")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--interval", type=int, default=120, help="seconds between polls")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    bot = Bot(args)
    bot.setup()
    while True:
        try:
            bot.run_once()
        except (subprocess.SubprocessError, OSError, ValueError) as e:   # GitHub or SSH hiccup: retry next poll
            if args.once:
                raise
            print(f"poll failed: {e} {getattr(e, 'stderr', '') or ''}"[:800], flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
