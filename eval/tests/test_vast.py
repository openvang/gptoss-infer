"""vast.ai rental lifecycle against a fake CLI: offer ranking, renting, reuse, idle release and the credit guard."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vast  # noqa: E402


def offer(oid, dph, perf_per_dollar):
    return {"id": oid, "dph_total": dph, "dlperf_per_dphtotal": perf_per_dollar, "geolocation": "XX"}


def inst(iid, status="running", label=vast.LABEL, port="40022"):
    return {"id": iid, "label": label, "actual_status": status, "public_ipaddr": "203.0.113.7",
            "ports": {"22/tcp": [{"HostPort": port}]} if port else {}}


class FakeCLI:
    """Answers Vast.call like the vastai CLI with --raw."""

    def __init__(self, offers=(), instances=(), credit=10.0, refuse=()):
        self.offers, self.instances, self.credit, self.refuse = list(offers), list(instances), credit, set(refuse)
        self.calls, self.next_id = [], 900

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("show", "instances"):
            for i in self.instances:                                  # a booting VM comes up after a few polls
                if i.get("boot_polls"):
                    i["boot_polls"] -= 1
                    i["actual_status"] = "loading" if i["boot_polls"] else "running"
            return [dict(i) for i in self.instances]
        if args[:2] == ("show", "user"):
            return {"credit": self.credit}
        if args[:2] == ("search", "offers"):
            assert "vms_enabled=true" in args[2] and "gpu_name=RTX_5090" in args[2]
            return list(self.offers)
        if args[:2] == ("create", "instance"):
            if int(args[2]) in self.refuse:
                raise subprocess.CalledProcessError(1, "vastai", stderr="no longer available")
            assert args[args.index("--image") + 1] == vast.IMAGE and args[args.index("--label") + 1] == vast.LABEL
            self.next_id += 1
            self.instances.append(inst(self.next_id))
            return {"success": True, "new_contract": self.next_id}
        if args[:2] == ("destroy", "instance"):
            assert args[3] == "-y"
            self.instances = [i for i in self.instances if i["id"] != int(args[2])]
            return {"success": True}
        raise AssertionError(args)


def make(cli, reachable=lambda iid: True, idle_minutes=20):
    t = [1000.0]
    v = vast.Vast("vastai", "k", lambda iid, target: reachable(iid), idle_minutes=idle_minutes,
                  clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s), boot_timeout=60)
    v.call = cli
    return v, t


def test_rank_prefers_performance_per_dollar_under_the_cap():
    offers = [offer(1, 0.55, 364), offer(2, 0.78, 251), offer(3, 0.47, 364), offer(4, 1.80, 110), offer(5, 0.40, None)]
    assert [o["id"] for o in vast.rank(offers, 1.0)] == [3, 1, 2, 5]


def test_rents_the_best_offer_and_reuses_it():
    cli = FakeCLI(offers=[offer(1, 0.78, 251), offer(2, 0.55, 364)])
    v, _ = make(cli)
    iid, target = v.target()
    assert iid == 901 and target == ("root@203.0.113.7", 40022)
    assert [c for c in cli.calls if c[0] == "create"][0][2] == "2"
    assert v.target() == (901, target) and sum(c[0] == "create" for c in cli.calls) == 1


def test_a_refused_offer_falls_to_the_next():
    cli = FakeCLI(offers=[offer(1, 0.55, 364), offer(2, 0.78, 251)], refuse={1})
    v, _ = make(cli)
    assert v.target()[0] == 901 and [c[2] for c in cli.calls if c[0] == "create"] == ["1", "2"]


def test_unreachable_or_stopped_instances_are_destroyed_and_replaced():
    cli = FakeCLI(offers=[offer(1, 0.55, 364)], instances=[inst(5, status="exited"), inst(6, label="other")])
    v, _ = make(cli, reachable=lambda iid: iid != 901)                 # the first rental never answers SSH
    with pytest.raises(RuntimeError):
        v.target()
    destroyed = [c[2] for c in cli.calls if c[0] == "destroy"]
    assert destroyed == ["5", "901"] and [i["id"] for i in cli.instances] == [6]   # never touches other labels


def test_credit_guard_and_price_cap():
    with pytest.raises(RuntimeError, match="credit"):
        make(FakeCLI(offers=[offer(1, 0.55, 364)], credit=1.0))[0].target()
    with pytest.raises(RuntimeError, match="no RTX 5090 VM offer"):
        make(FakeCLI(offers=[offer(1, 1.80, 110)]))[0].target()


def test_release_only_after_the_idle_period():
    cli = FakeCLI(instances=[inst(7)])
    v, t = make(cli, idle_minutes=20)
    v.used()
    t[0] += 19 * 60
    v.release_if_idle()
    assert [i["id"] for i in cli.instances] == [7]
    t[0] += 2 * 60
    v.release_if_idle()
    assert cli.instances == []


def test_read_key(tmp_path):
    (tmp_path / "a.env").write_text("OTHER=1\nVAST=abc123")
    (tmp_path / "b.env").write_text("VAST_API_KEY='xyz'\n")
    assert vast.read_key(tmp_path / "a.env") == "abc123" and vast.read_key(tmp_path / "b.env") == "xyz"
    (tmp_path / "c.env").write_text("NOPE=1\n")
    with pytest.raises(RuntimeError):
        vast.read_key(tmp_path / "c.env")


def test_a_booting_instance_is_waited_for_not_replaced():
    booting = dict(inst(8, status="loading"), boot_polls=3)
    cli = FakeCLI(offers=[offer(1, 0.55, 364)], instances=[booting])
    v, _ = make(cli)
    assert v.target()[0] == 8 and not any(c[0] in ("create", "destroy") for c in cli.calls)
