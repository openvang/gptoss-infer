#!/usr/bin/env python3
"""PR bot: evaluate pull requests in rounds on the GPU box, label each with its verdict, merge the round's largest
verified speedup, and close what cannot be scored.

Runs on a trusted host where `gh` is logged in as the maintainer account. The GPU box never receives GitHub
credentials: the bot ships commits to it as a git bundle over SSH, runs main's eval/run_eval.py there, and reads
back only the verdict JSON.

    python eval/bot.py --repo openvang/gptoss-infer --vast-env .env --key ~/.ssh/key [--once]       # vast.ai VMs
    python eval/bot.py --repo openvang/gptoss-infer --box root@HOST --port PORT --key ~/.ssh/key   # fixed box

Each poll is one round against the current main (CONTRIBUTING.md has the contributor-facing rules):
1. A contributor with more than policy.MAX_OPEN_PRS open PRs has the newest beyond that closed.
2. Each PR's changed files pick its lane (policy.lane): maintainer-owned paths get `eval:skipped`, PRs without
   runtime changes are left to a maintainer, and PRs mixing runtime and other files are asked to split. A
   contributor's runtime PR also needs the RTX 5090 proof (policy.proof): without the ticked box it is closed;
   without a gain in its table it gets `needs-benchmark`.
3. Every other PR without a verdict at its head on this main is merged onto main and evaluated on the box.
   `none` and `REJECT` close a contributor's PR.
4. The verified speedups on this main are ranked by conservative gain (policy.merge_order). The first that can
   merge gets `merge-first` and is squash-merged at its evaluated head; the others get `re-evaluate` and are
   measured again on the new main, so each is paid only for what it adds.
5. A contributor's PR that has waited on its author for policy.STALE_DAYS since the bot's last comment is closed.
While a PR waits on the bot, a status label shows where it is: `status:queued`, `status:node-starting` (a node is
being rented and set up) or `status:evaluating`. Drafts and PRs labelled `hold` are not evaluated or merged; `hold`
PRs are never closed. Org members and collaborators skip the proof check, the open-PR limit and every close.
"""
import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy  # noqa: E402
import vast  # noqa: E402

BOX_ROOT = "/data/gptoss-eval"
# What eval/box/provision.sh installs depends on these files; the box is provisioned again when they change.
PROVISION_INPUTS = ("eval/box/provision.sh", "eval/image/Dockerfile", "reference/weights.lock.json",
                    "reference/requirements.txt")
MARKER = "<!-- gptoss-eval head={head} base={base} label={label} score={score:.4f} -->"
MARKER_RE = re.compile(r"<!-- gptoss-eval head=(\w+) base=(\w+) label=([\w:-]+)(?: score=([\d.]+))? -->")
MEMBERS = {"OWNER", "MEMBER", "COLLABORATOR"}
# Marker labels that settle a head: once one is recorded, the head is not evaluated again on the same main.
SETTLED = set(policy.TIERS) | {"none", "REJECT", "skipped", "conflict"}
COLORS = {"XL": "0e8a16", "L": "2cbe4e", "M": "6ad17a", "S": "9fe2a6", "XS": "c9f1cd", "none": "bfbfbf",
          "REJECT": "d73a4a", "skipped": "fbca04"}
WORKFLOW_LABELS = {
    "merge-first": ("0e8a16", "the round's largest verified speedup: merged first"),
    "re-evaluate": ("1d76db", "verified speedup that lost its round: measured again on the new main"),
    "needs-rebase": ("fbca04", "does not merge cleanly onto main: rebase and push"),
    "needs-benchmark": ("fbca04", "RTX 5090 box ticked but no before/after gain in the description: not evaluated"),
}
# Where a PR waiting on the bot is, live. Not eval:* (SN74 pays on those).
STATUS_LABELS = {
    "status:queued": ("c5def5", "waiting for the bot's next round on the GPU"),
    "status:node-starting": ("d4c5f9", "an RTX 5090 node is being rented and set up for this round"),
    "status:evaluating": ("1d76db", "being built and measured on the RTX 5090 now"),
}
HOLD = ("d93f0b", "maintainer override: the bot does not evaluate, merge or close this PR")
MANAGED = set(policy.LABELS.values()) | set(WORKFLOW_LABELS) | set(STATUS_LABELS)   # the bot touches no other label


def run(cmd, input=None, timeout=900):
    """Run a command to completion. Every call is time-limited, so a hung gh, git or scp can't stall the bot."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True, input=input, timeout=timeout).stdout


def marker(head, base, label, score=0.0):
    return MARKER.format(head=head, base=base, label=label, score=score)


def parse_markers(bodies):
    """Markers from the bot's own comments. Only a comment's last line counts: the bot always appends its marker
    there, and anything above it may quote text a contributor controls (file names, failure reasons)."""
    found = []
    for body in bodies:
        lines = body.strip().splitlines()
        m = MARKER_RE.fullmatch(lines[-1].strip()) if lines else None
        if m:
            found.append(Marker(m[1], m[2], m[3], float(m[4] or 0)))
    return found


def quote_files(files, limit=10):
    """File names for a comment: contributor-controlled, so no markup and no long lists."""
    shown = ", ".join(f"`{re.sub(r'[`<>]', '?', f)}`" for f in files[:limit])
    return shown + (f" and {len(files) - limit} more" if len(files) > limit else "")


@dataclass
class Marker:
    head: str
    base: str
    label: str
    score: float


@dataclass
class Ready:
    """A PR verified as a speedup against the round's main, waiting to be ranked."""
    pr: dict
    tier: str
    score: float


class GitHub:
    """Everything the bot does on GitHub, through the gh CLI."""

    def __init__(self, repo, user=None):
        self.repo, self.user = repo, user

    def gh(self, *args, input=None):
        return run(["gh", *args], input=input)

    def login(self):
        return self.gh("api", "user", "--jq", ".login").strip()

    def ensure_account(self):
        """With a user set, act only as that account: if gh's active github.com account is another one, switch back."""
        if not self.user or self.login() == self.user:
            return
        print(f"gh: the active account is not {self.user}; switching to it", flush=True)
        self.gh("auth", "switch", "--hostname", "github.com", "--user", self.user)
        if self.login() != self.user:
            raise RuntimeError(f"gh could not switch to {self.user}")

    def main_sha(self):
        return self.gh("api", f"repos/{self.repo}/commits/main", "--jq", ".sha").strip()

    def open_prs(self):
        jq = ('.[] | {number, head: .head.sha, draft, base: .base.ref, labels: [.labels[].name], '
              'association: .author_association, author: .user.login, bot: (.user.type == "Bot"), '
              'body: (.body // ""), updated: .updated_at}')
        out = self.gh("api", f"repos/{self.repo}/pulls?state=open&per_page=100", "--paginate", "--jq", jq)
        return sorted((json.loads(line) for line in out.splitlines() if line.strip()), key=lambda p: p["number"])

    def pr(self, number):
        jq = "{state, head: .head.sha, draft, labels: [.labels[].name]}"
        return json.loads(self.gh("api", f"repos/{self.repo}/pulls/{number}", "--jq", jq))

    def files(self, number):
        out = self.gh("api", f"repos/{self.repo}/pulls/{number}/files?per_page=100", "--paginate", "--jq",
                      ".[].filename")
        return [line for line in out.splitlines() if line]

    def markers(self, number, login):
        # Only the bot's own comments count: anyone can post a comment that looks like a marker.
        jq = f'.[] | select(.user.login == "{login}") | .body | @json'
        out = self.gh("api", f"repos/{self.repo}/issues/{number}/comments?per_page=100", "--paginate", "--jq", jq)
        return parse_markers(json.loads(line) for line in out.splitlines() if line.strip())

    def comment(self, number, body):
        self.gh("pr", "comment", str(number), "--repo", self.repo, "--body-file", "-", input=body)

    def close(self, number):
        self.gh("pr", "close", str(number), "--repo", self.repo)

    def edit_labels(self, number, add, remove):
        args = ["pr", "edit", str(number), "--repo", self.repo]
        for name in add:
            args += ["--add-label", name]
        for name in remove:
            args += ["--remove-label", name]
        self.gh(*args)

    def merge(self, number, head):
        self.gh("pr", "merge", str(number), "--repo", self.repo, "--squash", "--match-head-commit", head)

    def ensure_labels(self, wanted):
        existing = {l["name"] for l in json.loads(self.gh("label", "list", "--repo", self.repo, "--json", "name",
                                                           "--limit", "500"))}
        for name, (color, description) in wanted.items():
            if name not in existing:
                self.gh("label", "create", name, "--repo", self.repo, "--color", color, "--description", description)


class Box:
    """The GPU box. Builds each candidate locally (the PR merged onto main), ships it with main, prepares the box with
    main's eval/box/provision.sh when what it installs has changed, and runs main's eval/run_eval.py there. The box
    is a fixed SSH host (--box) or a vast.ai VM rented for the work (--vast-env, eval/vast.py)."""

    def __init__(self, args, vast=None):
        self.a = args
        self.vast = vast
        self.work = Path(args.workdir).expanduser()
        self.mirror = self.work / "mirror"
        self.key = str(Path(args.key).expanduser())
        self.golden = Path(args.golden).expanduser()
        self.node = None

    def git(self, *args, cwd=None):
        return run(["git", "-C", str(cwd or self.mirror), *args]).strip()

    def setup(self):
        self.work.mkdir(parents=True, exist_ok=True)
        if not self.mirror.exists():
            run(["gh", "repo", "clone", self.a.repo, str(self.mirror), "--", "-q"])
            # Local merge commits only (never pushed) need an identity.
            self.git("config", "user.name", "gptoss-eval-bot")
            self.git("config", "user.email", "gptoss-eval-bot@localhost")

    # -- SSH to the current box: a rented VM gets its own known_hosts file, trusted on first use ------------------

    def _opts(self, iid):
        # Only the bot's key: an agent offering other keys first could hit the server's MaxAuthTries.
        opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "IdentitiesOnly=yes"]
        if iid is not None:
            opts += ["-o", f"UserKnownHostsFile={self.work / f'known_hosts-{iid}'}", "-o", "StrictHostKeyChecking=accept-new"]
        return opts

    def ssh(self, iid, target):
        return ["ssh", "-i", self.key, "-p", str(target[1]), *self._opts(iid), target[0]]

    def scp(self, iid, target):
        return ["scp", "-q", "-i", self.key, "-P", str(target[1]), *self._opts(iid)]

    def ssh_ok(self, iid, target):
        try:
            return subprocess.run(self.ssh(iid, target) + ["true"], capture_output=True, stdin=subprocess.DEVNULL,
                                  timeout=90).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def target(self):
        return self.vast.target() if self.vast else (None, (self.a.box, self.a.port))

    def needs_node(self):
        """True when this round has to rent a node before it can evaluate."""
        return self.vast is not None and not self.vast.running()

    def prepare(self, main):
        """Get the round's node (renting one if needed) and provision it with main's harness."""
        self.node = self.target()
        try:
            self._remote(*self.node, main, None)
        finally:
            if self.vast:
                self.vast.used()

    def after_round(self):
        self.node = None
        if self.vast:
            self.vast.release_if_idle()

    # -- one evaluation ----------------------------------------------------------------------------------------------

    def candidate(self, pr, main):
        """Merge the PR head onto main locally; returns the merged commit, or None on conflict."""
        n = pr["number"]
        self.git("fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main",
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

    def evaluate(self, main, cand):
        try:
            return self._remote(*(self.node or self.target()), main, cand)
        finally:
            if self.vast:
                self.vast.used()

    def _remote(self, iid, target, main, cand):
        """Ship main (and the candidate), provision the box if main's provisioning inputs changed, then evaluate the
        candidate. Without a candidate, only provisions."""
        host, ssh, scp = target[0], self.ssh(iid, target), self.scp(iid, target)
        refs = ["refs/eval/base"] + (["refs/eval/cand"] if cand else [])
        self.git("update-ref", "refs/eval/base", main)
        if cand:
            self.git("update-ref", "refs/eval/cand", cand)
        golden = f"{BOX_ROOT}/goldens/{self.golden.name}"            # a release artifact, checked by run_eval.py
        if subprocess.run(ssh + [f"mkdir -p {BOX_ROOT}/goldens && test -f {golden}"], capture_output=True,
                          stdin=subprocess.DEVNULL, timeout=300).returncode:
            run(scp + [str(self.golden), f"{host}:{golden}"])
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "eval.bundle"
            self.git("bundle", "create", str(bundle), *refs)
            run(scp + [str(bundle), f"{host}:{BOX_ROOT}/incoming.bundle"])
        q = shlex.quote
        stamp_inputs = " ".join(f"harness/{f}" for f in PROVISION_INPUTS)
        remote = "\n".join([
            "set -e",
            f"cd {BOX_ROOT}",
            "[ -d repo ] || git init -q repo",
            f"git -C repo fetch -q {BOX_ROOT}/incoming.bundle '+refs/eval/*:refs/eval/*'",
            f"rm -rf harness && mkdir harness && git -C repo archive {q(main)} | tar -x -C harness",
            # Provision once per box, and again whenever main changes what provisioning installs.
            f"stamp=$(cat {stamp_inputs} | sha256sum | cut -c1-16)",
            'if [ "$(cat provisioned 2>/dev/null)" != "$stamp" ]; then',
            "  timeout 3600 bash harness/eval/box/provision.sh > provision.log 2>&1 || { tail -n 30 provision.log >&2; exit 3; }",
            '  echo "$stamp" > provisioned',
            "fi",
        ] + ([f"{self.a.box_python} harness/eval/run_eval.py --repo repo --base {q(main)} --cand {q(cand)} "
              f"--models {self.a.box_models} --goldens {BOX_ROOT}/goldens --work {BOX_ROOT}/runs --pairs {self.a.pairs}"]
             if cand else []))
        proc = subprocess.run(ssh + [remote], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=4 * 3600)
        if not cand:
            if proc.returncode:
                raise RuntimeError(f"provisioning failed on the box (exit {proc.returncode}): {proc.stderr[-1500:]}")
            return None
        lines = proc.stdout.strip().splitlines()
        if not lines or not lines[-1].endswith("verdict.json"):
            raise RuntimeError(f"eval failed on the box (exit {proc.returncode}): {proc.stderr[-1500:]}")
        with tempfile.TemporaryDirectory() as d:
            run(scp + [f"{host}:{lines[-1]}", f"{d}/verdict.json"])
            return json.loads(Path(d, "verdict.json").read_text())


class Bot:
    def __init__(self, github, box, now=None):
        self.gh = github
        self.box = box
        self.now = now or (lambda: dt.datetime.now(dt.timezone.utc))
        self.login = None
        self.guide = f"https://github.com/{github.repo}/blob/main/CONTRIBUTING.md"

    def setup(self):
        self.gh.ensure_account()
        self.login = self.gh.login()
        self.box.setup()
        wanted = {name: (COLORS[key], f"gptoss eval verdict: {key}") for key, name in policy.LABELS.items()}
        wanted.update(WORKFLOW_LABELS)
        wanted.update(STATUS_LABELS)
        wanted["hold"] = HOLD
        self.gh.ensure_labels(wanted)

    # -- GitHub writes, recorded on the PR dict so the round knows what it changed ---------------------------------

    def sync_labels(self, pr, desired):
        present = set(pr["labels"]) & MANAGED
        add, remove = sorted(desired - present), sorted(present - desired)
        if add or remove:
            self.gh.edit_labels(pr["number"], add, remove)
            pr["labels"] = sorted((set(pr["labels"]) - set(remove)) | set(add))
            pr["touched"] = True

    def set_status(self, pr, status):
        """Replace the PR's status label (None clears it), keeping its other managed labels."""
        keep = (set(pr["labels"]) & MANAGED) - set(STATUS_LABELS)
        self.sync_labels(pr, keep | ({status} if status else set()))

    def post(self, pr, text, state, main, score=0.0):
        self.gh.comment(pr["number"], f"{text}\n{marker(pr['head'], main, state, score)}")
        pr["touched"] = True

    def close(self, pr, text, state, main, drafts=False):
        """Comment and close a contributor's PR, unless it changed since the round read it."""
        if pr["association"] in MEMBERS:
            return False
        fresh = self.gh.pr(pr["number"])
        if (fresh["state"] != "open" or fresh["head"] != pr["head"] or "hold" in fresh["labels"]
                or (fresh["draft"] and not drafts)):
            return False
        self.post(pr, text, state, main)
        self.gh.close(pr["number"])
        print(f"  #{pr['number']}: closed ({state})", flush=True)
        return True

    # -- one round ------------------------------------------------------------------------------------------------

    def run_once(self):
        self.gh.ensure_account()
        main = self.gh.main_sha()
        prs = self.enforce_cap([p for p in self.gh.open_prs() if p["base"] == "main"], main)
        queue, ready, waiting = [], [], []
        for pr in prs:
            if pr["draft"] or "hold" in pr["labels"]:
                self.isolated(pr, lambda: self.set_status(pr, None))      # no longer waiting on the bot
                continue
            state = self.isolated(pr, lambda: self.triage(
                pr, main, [m for m in self.gh.markers(pr["number"], self.login) if m.head == pr["head"]]))
            if state == "evaluate":
                queue.append(pr)
            elif state == "author":
                waiting.append(pr)
            elif isinstance(state, Ready):
                ready.append(state)
        merged = [(pr, cand) for pr in queue if (cand := self.isolated(pr, lambda: self.candidate(pr, main)))]
        if merged and self.prepare(merged, main):
            for pr, cand in merged:
                result = self.isolated(pr, lambda: self.evaluate(pr, main, cand))
                if result:
                    ready.append(result)
        self.merge_round(ready, main)
        self.close_stale(waiting, main)
        self.box.after_round()

    @staticmethod
    def isolated(pr, step):
        """One PR's GitHub failure must not stop the round for the others."""
        try:
            return step()
        except subprocess.SubprocessError as e:
            print(f"  #{pr['number']}: {e} {(e.stderr or '').strip()[:300]}", flush=True)
            return None

    def enforce_cap(self, prs, main):
        by_author = {}
        for pr in prs:
            if not pr["bot"]:
                by_author.setdefault(pr["author"], []).append(pr)
        closed = set()
        for author, own in by_author.items():
            if len(own) <= policy.MAX_OPEN_PRS or any(p["association"] in MEMBERS for p in own):
                continue
            for pr in own[policy.MAX_OPEN_PRS:]:                        # open_prs() is sorted oldest first
                text = (f"**gptoss eval: closed (open-PR limit)**\n\n@{author} has {len(own)} open pull requests; "
                        f"the limit is {policy.MAX_OPEN_PRS}, so the newest beyond it are closed. Get some merged "
                        f"or closed, then reopen this one. See the [contribution rules]({self.guide}).")
                if self.close(pr, text, "cap", main, drafts=True):
                    closed.add(pr["number"])
        return [p for p in prs if p["number"] not in closed]

    def triage(self, pr, main, marks):
        """Lane and proof checks. Returns "evaluate", a Ready, "author" (waiting on the author since an earlier
        round's comment) or None (nothing waits on the author)."""
        last = marks[-1].label if marks else None

        def tell(state, text, labels):
            self.sync_labels(pr, labels)
            if last != state:
                self.post(pr, text, state, main)
                return None
            return "author"

        files = self.gh.files(pr["number"])
        lane = policy.lane(files)
        if lane == "protected":
            touched = quote_files([f for f in files if f.startswith(policy.PROTECTED)])
            tell("skipped", f"**gptoss eval: `{policy.LABELS['skipped']}`**\n\nThis PR changes maintainer-owned "
                            f"paths ({touched}), which define how PRs are measured, so it is not evaluated or "
                            f"scored. A maintainer reviews it.", {policy.LABELS["skipped"]})
            return None
        if lane == "manual":
            tell("manual", "**gptoss eval: not scored**\n\nThis PR changes no runtime code (`runtime/`, "
                           "`CMakeLists.txt`), so it is not measured. A maintainer reviews it.", set())
            return None
        if lane == "mixed":
            other = quote_files([f for f in files if not f.startswith(policy.SCORED)])
            return tell("mixed", f"**gptoss eval: not evaluated**\n\nA scored PR changes only `runtime/` and "
                                 f"`CMakeLists.txt`. Move the other files ({other}) to a separate PR and push.", set())
        if pr["association"] not in MEMBERS:
            proof = policy.proof(pr["body"])
            if proof == "unticked":
                self.sync_labels(pr, set())
                self.close(pr, f"**gptoss eval: closed (no RTX 5090 proof)**\n\nRuntime changes are evaluated only "
                               f"when the description ticks `- [x] Tested on RTX 5090` and fills the before/after "
                               f"table with your `gptoss-bench` numbers ([how]({self.guide})). Edit the description, "
                               f"then reopen.", "unticked", main)
                return None
            if proof == "no-gain":
                return tell("needs-benchmark", "**gptoss eval: `needs-benchmark`**\n\nThe RTX 5090 box is ticked, "
                                               "but the before/after table has no column where this PR is faster. "
                                               "Fill in your `gptoss-bench` numbers; the description is re-read "
                                               "every round.", {"needs-benchmark"})
        settled = [m for m in marks if m.label in SETTLED]
        if not settled:
            self.sync_labels(pr, {"status:queued"})                      # labels from an older head no longer apply
            return "evaluate"
        v = settled[-1]
        if v.label == "conflict":
            self.sync_labels(pr, {"needs-rebase"})
            return "author"
        if v.label in policy.TIERS:
            if v.base == main:
                return Ready(pr, v.label, v.score)
            self.set_status(pr, "status:queued")
            return "evaluate"
        if v.label in ("none", "REJECT"):                                # reopened without a new commit
            self.close(pr, f"Closed: this commit already has its verdict (`{policy.LABELS[v.label]}`). Push a new "
                           f"commit and reopen to be evaluated again.", "closed", main)
        return None

    def candidate(self, pr, main):
        """The PR merged onto main (locally), or None after asking for a rebase."""
        cand = self.box.candidate(pr, main)
        if cand is None:
            self.sync_labels(pr, {"needs-rebase"})
            self.post(pr, f"**gptoss eval: `needs-rebase`**\n\nThis PR does not merge cleanly onto main "
                          f"(`{main[:8]}`). Rebase and push; the new head is evaluated in the next round.",
                      "conflict", main)
            print(f"  #{pr['number']}: conflict", flush=True)
        return cand

    def prepare(self, merged, main):
        """A ready box for this round's evaluations; the waiting PRs show it while a node is rented and set up."""
        if self.box.needs_node():
            for pr, _ in merged:
                self.isolated(pr, lambda: self.set_status(pr, "status:node-starting"))
        try:
            self.box.prepare(main)
            return True
        except Exception as e:                                           # no offer, no credit, setup failed: retry
            print(f"box not ready: {e}", flush=True)
            for pr, _ in merged:
                self.isolated(pr, lambda: self.set_status(pr, "status:queued"))
            return False

    def evaluate(self, pr, main, cand):
        n, head = pr["number"], pr["head"]
        print(f"evaluating #{n} {head[:8]} onto main {main[:8]}", flush=True)
        self.set_status(pr, "status:evaluating")
        try:
            verdict = self.box.evaluate(main, cand)
        except Exception as e:                                           # infrastructure problem: retry next round
            print(f"  #{n}: {e}", flush=True)
            self.set_status(pr, "status:queued")
            return None
        label = verdict["label"]
        if label == "error":
            print(f"  #{n}: infra error {verdict.get('reasons')}", flush=True)
            self.set_status(pr, "status:queued")
            return None
        score = policy.speedup_score(verdict["axes"]) if label in policy.TIERS else 0.0
        self.post(pr, self.verdict_text(verdict, main), label, main, score)
        self.sync_labels(pr, {policy.LABELS[label]})
        print(f"  #{n}: {label}", flush=True)
        if label in policy.TIERS:
            return Ready(pr, label, score)
        if label in ("none", "REJECT"):
            self.close(pr, f"Closed: `{policy.LABELS[label]}`, no verified speedup (measurements above). Push a "
                           f"new commit and reopen to be evaluated again.", "closed", main)
        return None

    def merge_round(self, ready, main):
        if not ready:
            return
        if self.gh.main_sha() != main:
            print("main moved during the round: verified PRs are measured again next round", flush=True)
            return
        by_number = {r.pr["number"]: r for r in ready}
        order = policy.merge_order([(r.pr["number"], r.score) for r in ready])
        for i, n in enumerate(order):
            r = by_number[n]
            fresh = self.gh.pr(n)
            if fresh["state"] != "open" or fresh["head"] != r.pr["head"] or fresh["draft"] or "hold" in fresh["labels"]:
                continue
            self.sync_labels(r.pr, {policy.LABELS[r.tier], "merge-first"})
            try:
                self.gh.merge(n, r.pr["head"])
            except subprocess.SubprocessError as e:
                print(f"  #{n}: merge refused: {(e.stderr or '').strip()[:300]}", flush=True)
                self.sync_labels(r.pr, {policy.LABELS[r.tier]})
                continue
            print(f"  #{n}: merged ({r.tier})", flush=True)
            # PRs ranked above the winner (held, re-pushed or refused) are simply measured again next round.
            for loser in (by_number[m] for m in order[i + 1:]):
                self.sync_labels(loser.pr, {policy.LABELS[loser.tier], "re-evaluate"})
                self.post(loser.pr, f"**gptoss eval: `re-evaluate`**\n\nVerified `{policy.LABELS[loser.tier]}`, "
                                    f"but #{n} had the larger gain this round and merged first. This PR is measured "
                                    f"again on the new main, so only its gain on top of #{n} counts.",
                          "re-evaluate", main, loser.score)
            return

    def close_stale(self, waiting, main):
        limit = dt.timedelta(days=policy.STALE_DAYS)
        for pr in waiting:
            if pr.get("touched"):
                continue
            updated = dt.datetime.fromisoformat(pr["updated"].replace("Z", "+00:00"))
            if self.now() - updated >= limit:
                self.close(pr, f"**gptoss eval: closed (inactive)**\n\nThis PR waited on its author for "
                               f"{policy.STALE_DAYS} days after the bot's last comment. Push a commit and reopen it "
                               f"to continue.", "stale", main)

    def verdict_text(self, verdict, main):
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
                      f"`{verdict['run']}`</sub>"]
        return "\n".join(lines)


def code_changed(checkout):
    """True when origin/main's eval/ differs from the checkout the bot is running from."""
    try:
        run(["git", "-C", str(checkout), "fetch", "-q", "origin", "main"], timeout=120)
    except (subprocess.SubprocessError, OSError):
        return False
    return subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "FETCH_HEAD", "--", "eval/"],
                          capture_output=True).returncode == 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gh-user", help="the account the bot acts as; gh is switched back to it if another is active")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--box", help="user@host of a fixed GPU box")
    where.add_argument("--vast-env", help="env file with VAST_API_KEY (or VAST): rent RTX 5090 VMs on vast.ai")
    ap.add_argument("--port", type=int, default=22, help="SSH port of --box")
    ap.add_argument("--key", required=True, help="SSH private key for the box")
    ap.add_argument("--vast-cli", default="vastai", help="the vast.ai CLI (pip install vastai)")
    ap.add_argument("--vast-max-dph", type=float, default=1.0, help="highest $/hour to rent at")
    ap.add_argument("--vast-idle-minutes", type=int, default=7, help="destroy the VM after this long without work")
    ap.add_argument("--box-python", default="/data/venv/bin/python", help="the judge's Python (eval/box/provision.sh)")
    ap.add_argument("--box-models", default="/data/models", help="directory holding gpt-oss-20b on the box")
    ap.add_argument("--golden", default=str(HERE.parent / "reference/goldens/golden_v1.safetensors"),
                    help="the golden release artifact, uploaded to a box that lacks it")
    ap.add_argument("--workdir", default="~/.cache/gptoss-eval-bot")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--interval", type=int, default=120, help="seconds between rounds")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--restart-on-update", action="store_true",
                    help="exit (for the service manager to restart it on the new code) once main changes eval/")
    args = ap.parse_args()
    box = Box(args)
    if args.vast_env:
        box.vast = vast.Vast(args.vast_cli, vast.read_key(Path(args.vast_env).expanduser()), box.ssh_ok,
                             max_dph=args.vast_max_dph, idle_minutes=args.vast_idle_minutes)
    bot = Bot(GitHub(args.repo, args.gh_user), box)
    bot.setup()
    while True:
        try:
            bot.run_once()
        except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as e:   # GitHub, SSH or vast.ai hiccup
            if args.once:
                raise
            print(f"poll failed: {e} {getattr(e, 'stderr', '') or ''}"[:800], flush=True)
        if args.once:
            break
        if args.restart_on_update and code_changed(HERE.parent):
            print("eval/ changed on main: exiting so the service restarts on the new code", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
