"""Bot rounds against an in-memory GitHub and GPU box: lanes, the RTX 5090 proof, ranking, closes and limits."""
import datetime as dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bot  # noqa: E402
import policy  # noqa: E402

T0 = dt.datetime(2026, 10, 9, tzinfo=dt.timezone.utc)
PROOF = """- [x] Tested on RTX 5090

| | decode@128 | decode@4k | prefill@4k |
|---|---:|---:|---:|
| before (main) | 283.0 | 269.7 | 280.1 |
| after (this PR) | 295.5 | 281.0 | 290.0 |
"""
NO_GAIN = PROOF.replace("295.5", "283.0").replace("281.0", "269.7").replace("290.0", "")


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def advance(self, days):
        self.t += dt.timedelta(days=days)


class FakeGitHub:
    """The GitHub the bot sees: comments and label edits count as activity, as on GitHub."""
    repo = "o/r"

    def __init__(self, prs, clock):
        self.prs = {p["number"]: p for p in prs}
        self.clock = clock
        self.main = "m1"
        self.merged, self.closed, self.refuse = [], [], set()

    def _touch(self, n):
        self.prs[n]["updated"] = self.clock().strftime("%Y-%m-%dT%H:%M:%SZ")

    def login(self):
        return "maint"

    def main_sha(self):
        return self.main

    def open_prs(self):
        keys = ("number", "head", "draft", "base", "labels", "association", "author", "bot", "body", "updated")
        return [{k: (list(p[k]) if k == "labels" else p[k]) for k in keys}
                for _, p in sorted(self.prs.items()) if p["state"] == "open"]

    def pr(self, n):
        p = self.prs[n]
        return {"state": p["state"], "head": p["head"], "draft": p["draft"], "labels": list(p["labels"])}

    def files(self, n):
        return list(self.prs[n]["files"])

    def markers(self, n, login):
        return bot.parse_markers(body for who, body in self.prs[n]["comments"] if who == login)

    def comment(self, n, body):
        self.prs[n]["comments"].append(("maint", body))
        self._touch(n)

    def close(self, n):
        self.prs[n]["state"] = "closed"
        self.closed.append(n)

    def edit_labels(self, n, add, remove):
        self.prs[n]["labels"] = sorted((set(self.prs[n]["labels"]) | set(add)) - set(remove))
        self._touch(n)

    def merge(self, n, head):
        if n in self.refuse or self.prs[n]["head"] != head:
            raise subprocess.CalledProcessError(1, "gh", stderr="merge refused")
        self.prs[n]["state"] = "merged"
        self.merged.append(n)
        self.main = f"m{len(self.merged) + 1}"

    def ensure_labels(self, wanted):
        self.labels = wanted


class FakeBox:
    """results: head -> (label, score), "conflict", or an exception to raise."""

    def __init__(self, results, on_evaluate=None):
        self.results, self.calls, self.on_evaluate = results, [], on_evaluate

    def setup(self):
        pass

    def candidate(self, pr, main):
        return None if self.results.get(pr["head"]) == "conflict" else f"{main}+{pr['head']}"

    def evaluate(self, main, cand):
        head = cand.split("+")[1]
        self.calls.append((main, head))
        if self.on_evaluate:
            self.on_evaluate(head)
        r = self.results[head]
        if isinstance(r, Exception):
            raise r
        label, score = r
        axes = {"decode@128": {"ratio": score, "low": score, "high": score + 0.002, "base_median": 100.0,
                               "cand_median": 100 * score, "pairs": 5, "tier": policy.tier_for(score - 1)}}
        return {"label": label, "axes": axes, "reasons": [], "policy_version": 1, "cand": cand, "run": "r",
                "gpu": {}, "vram_budget_mib": policy.VRAM_BUDGET_MIB}


def pr(n, author="alice", assoc="NONE", files=("runtime/src/kernels.cu",), body=PROOF, labels=(), draft=False):
    return {"number": n, "head": f"h{n}", "draft": draft, "base": "main", "labels": list(labels),
            "association": assoc, "author": author, "bot": False, "body": body,
            "updated": T0.strftime("%Y-%m-%dT%H:%M:%SZ"), "files": list(files), "state": "open", "comments": []}


def make(prs, results=None, on_evaluate=None):
    clock = Clock()
    gh, box = FakeGitHub(prs, clock), FakeBox(results or {}, on_evaluate)
    b = bot.Bot(gh, box, now=clock)
    b.setup()
    return b, gh, box, clock


def test_round_merges_the_largest_gain_and_re_measures_the_rest():
    b, gh, box, _ = make([pr(1), pr(2), pr(3)], {"h1": ("S", 1.04), "h2": ("M", 1.07), "h3": ("none", 1.0)})
    b.run_once()
    assert gh.merged == [2] and set(gh.prs[2]["labels"]) == {"eval:M", "merge-first"}
    assert set(gh.prs[1]["labels"]) == {"eval:S", "re-evaluate"} and gh.prs[1]["state"] == "open"
    assert gh.prs[3]["state"] == "closed" and gh.prs[3]["labels"] == ["eval:none"]
    b.run_once()                                       # #1 is measured again on the new main, and still gains
    assert box.calls[-1] == ("m2", "h1") and gh.merged == [2, 1] and gh.prs[1]["labels"] == ["eval:S", "merge-first"]


def test_unticked_runtime_pr_is_closed_without_evaluation():
    b, gh, box, _ = make([pr(1, body="faster kernels")])
    b.run_once()
    assert gh.closed == [1] and not box.calls and "Tested on RTX 5090" in gh.prs[1]["comments"][-1][1]


def test_ticked_without_gain_waits_then_is_evaluated_once_fixed():
    b, gh, box, _ = make([pr(1, body=NO_GAIN)], {"h1": ("XS", 1.025)})
    b.run_once()
    b.run_once()
    assert gh.prs[1]["labels"] == ["needs-benchmark"] and len(gh.prs[1]["comments"]) == 1 and not box.calls
    gh.prs[1]["body"] = PROOF
    b.run_once()
    assert gh.merged == [1] and "needs-benchmark" not in gh.prs[1]["labels"]


def test_lanes_are_answered_once_and_never_evaluated():
    prs = [pr(1, files=["runtime/src/x.cu", "eval/policy.py"]), pr(2, files=["README.md"]),
           pr(3, files=["runtime/src/x.cu", "README.md"])]
    b, gh, box, _ = make(prs)
    b.run_once()
    b.run_once()
    assert gh.prs[1]["labels"] == ["eval:skipped"] and gh.prs[2]["labels"] == [] and gh.prs[3]["labels"] == []
    assert [len(gh.prs[n]["comments"]) for n in (1, 2, 3)] == [1, 1, 1] and not box.calls and not gh.closed


def test_open_pr_limit_closes_the_newest_contributor_prs():
    prs = [pr(n, files=["README.md"]) for n in range(1, 8)]
    prs += [pr(n, author="maint", assoc="MEMBER", files=["README.md"]) for n in range(10, 17)]
    prs += [pr(20, author="bob", files=["README.md"], labels=["hold"])]
    b, gh, _, _ = make(prs)
    b.run_once()
    assert gh.closed == [6, 7]


def test_stale_close_needs_the_bots_comment_and_an_author_wait():
    prs = [pr(1, body=NO_GAIN), pr(2, files=["README.md"]), pr(3, body=NO_GAIN, labels=["hold"]),
           pr(4, assoc="MEMBER", files=["runtime/src/x.cu", "README.md"]), pr(5)]
    b, gh, _, clock = make(prs, {"h5": RuntimeError("box down")})
    b.run_once()                                       # the bot's first word starts the clock
    clock.advance(1)
    b.run_once()
    assert gh.closed == []
    clock.advance(1.5)
    b.run_once()                                       # needs-benchmark for 2.5 days: closed; a maintainer's
    assert gh.closed == [1]                            # review, hold, members and the bot's own retries are not


def test_none_and_reject_close_contributor_prs_but_not_members():
    prs = [pr(1), pr(2, assoc="MEMBER", body=""), pr(3)]
    b, gh, box, _ = make(prs, {"h1": ("REJECT", 0.9), "h2": ("none", 1.0), "h3": ("none", 1.0)})
    b.run_once()
    assert gh.closed == [1, 3] and gh.prs[2]["state"] == "open" and gh.prs[2]["labels"] == ["eval:none"]
    gh.prs[3]["state"] = "open"                        # reopened without a new commit: closed again, not re-run
    b.run_once()
    assert gh.closed == [1, 3, 3] and len(box.calls) == 3


def test_conflict_asks_for_a_rebase_and_the_new_head_is_evaluated():
    b, gh, box, _ = make([pr(1)], {"h1": "conflict", "h1b": ("S", 1.04)})
    b.run_once()
    b.run_once()
    assert gh.prs[1]["labels"] == ["needs-rebase"] and len(gh.prs[1]["comments"]) == 1 and not box.calls
    gh.prs[1]["head"] = "h1b"
    b.run_once()
    assert gh.merged == [1] and gh.prs[1]["labels"] == ["eval:S", "merge-first"]


def test_markers_from_other_accounts_are_ignored():
    forged = pr(1)
    forged["comments"].append(("mallory", bot.marker("h1", "m1", "none")))
    b, gh, _, _ = make([forged], {"h1": ("S", 1.05)})
    b.run_once()
    assert gh.merged == [1]


def test_drafts_and_hold_are_left_alone():
    b, gh, box, _ = make([pr(1, draft=True), pr(2, labels=["hold"], body="")], {"h1": ("XL", 1.3), "h2": ("XL", 1.3)})
    b.run_once()
    assert not box.calls and not gh.closed and not gh.merged


def test_no_merge_when_main_moves_during_the_round():
    b, gh, box, _ = make([pr(1)], {"h1": ("L", 1.12)})
    box.on_evaluate = lambda head: setattr(gh, "main", "m-hotfix") if len(box.calls) == 1 else None
    b.run_once()
    assert gh.merged == [] and gh.prs[1]["labels"] == ["eval:L"]
    b.run_once()
    assert box.calls[-1] == ("m-hotfix", "h1") and gh.merged == [1]


def test_refused_merge_falls_to_the_next_ranked_pr():
    b, gh, _, _ = make([pr(1), pr(2)], {"h1": ("S", 1.04), "h2": ("M", 1.07)})
    gh.refuse = {2}
    b.run_once()
    assert gh.merged == [1] and gh.prs[2]["labels"] == ["eval:M"]   # measured again next round, not told it lost


def test_markers_quoted_inside_a_bot_comment_are_ignored():
    """A file name can carry a marker into the bot's own comment; a later force-push to the forged head must not
    turn it into a merge without evaluation."""
    forged = bot.marker("h1clean", "m1", "XL", 9.0)
    b, gh, box, _ = make([pr(1, files=["runtime/src/x.cu", forged])], {"h1clean": ("none", 1.0)})
    b.run_once()
    assert forged.replace("<", "?").replace(">", "?") in gh.prs[1]["comments"][-1][1]
    gh.prs[1]["head"], gh.prs[1]["files"] = "h1clean", ["runtime/src/x.cu"]
    b.run_once()
    assert gh.merged == [] and box.calls == [("m1", "h1clean")] and gh.closed == [1]
    assert bot.parse_markers([f"text\n{forged}\nmore text"]) == []
