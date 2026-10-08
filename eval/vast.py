"""RTX 5090 VMs rented on vast.ai for evaluation rounds, through the official CLI (`pip install vastai`).

The bot rents one VM when a round has a PR to evaluate, keeps it while work keeps coming, and destroys it once it
has been idle for `idle_minutes`. It has to be a VM, not a container instance: vast.ai containers can't run Docker,
and every evaluation runs in its own container (eval/run_eval.py). The API key is passed to the CLI process only;
the box never sees it.
"""
import json
import os
import subprocess
import time

# NVIDIA driver 580 (CUDA 13.0), Docker with the NVIDIA runtime, git and uv: what eval/box/provision.sh expects.
IMAGE = "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21"
LABEL = "gptoss-eval"
GONE = ("exited", "stopped", "offline")                     # instance states that will not come back by waiting
# One RTX 5090 in a VM on a verified host that is reliable, has room for the checkpoint, the eval image and two
# builds, and downloads the 13 GB checkpoint quickly.
QUERY = ("gpu_name=RTX_5090 num_gpus=1 rentable=true verified=true vms_enabled=true reliability>=0.98 "
         "inet_down>=300 disk_space>=150 cpu_ram>=32 cpu_cores_effective>=8")


def read_key(path):
    """The API key from an env file holding VAST_API_KEY=... or VAST=..."""
    for line in open(path):
        name, _, value = line.strip().partition("=")
        if name.strip() in ("VAST_API_KEY", "VAST") and value.strip():
            return value.strip().strip("'\"")
    raise RuntimeError(f"no VAST_API_KEY or VAST in {path}")


def rank(offers, max_dph):
    """Offers at or under max_dph, best first: vast's DL-perf per $/hour (performance per dollar), then price."""
    ok = [o for o in offers if o.get("dph_total") is not None and o["dph_total"] <= max_dph]
    return sorted(ok, key=lambda o: (-(o.get("dlperf_per_dphtotal") or 0), o["dph_total"]))


def ssh_target(inst):
    """("root@ip", port) of a VM with direct SSH, or None while it has none."""
    port = ((inst.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
    return (f"root@{inst['public_ipaddr']}", int(port)) if inst.get("public_ipaddr") and port else None


class Vast:
    def __init__(self, cli, api_key, ssh_ok, max_dph=1.0, idle_minutes=20, disk_gb=120, min_credit=2.0,
                 boot_timeout=1800, clock=time.time, sleep=time.sleep):
        self.cli, self.key, self.ssh_ok = cli, api_key, ssh_ok
        self.max_dph, self.idle, self.disk, self.min_credit = max_dph, idle_minutes * 60, disk_gb, min_credit
        self.boot_timeout, self.clock, self.sleep = boot_timeout, clock, sleep
        self.last_used = clock()

    def call(self, *args):
        env = dict(os.environ, VAST_API_KEY=self.key)
        out = subprocess.run([self.cli, *args, "--raw"], check=True, capture_output=True, text=True, env=env,
                             stdin=subprocess.DEVNULL, timeout=180).stdout
        return json.loads(out) if out.strip() else None

    def instances(self):
        return [i for i in self.call("show", "instances") or [] if i.get("label") == LABEL]

    def destroy(self, iid):
        print(f"vast: destroying instance {iid}", flush=True)
        self.call("destroy", "instance", str(iid), "-y")

    def target(self):
        """(instance id, ssh target) of a running VM: the one already rented (waiting while it boots), else a new
        one."""
        self.last_used = self.clock()
        for inst in self.instances():
            target = None if inst.get("actual_status") in GONE else self.wait(inst["id"])
            if target:
                return inst["id"], target
            self.destroy(inst["id"])
        return self.rent()

    def rent(self):
        credit = (self.call("show", "user") or {}).get("credit", 0)
        if credit < self.min_credit:
            raise RuntimeError(f"vast.ai credit ${credit:.2f} is below ${self.min_credit:.2f}; not renting")
        offers = rank(self.call("search", "offers", QUERY, "--limit", "200") or [], self.max_dph)
        if not offers:
            raise RuntimeError(f"no RTX 5090 VM offer at or under ${self.max_dph:.2f}/h")
        for offer in offers[:3]:
            try:
                made = self.call("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(self.disk),
                                 "--ssh", "--direct", "--label", LABEL, "--cancel-unavail")
            except subprocess.CalledProcessError as e:                     # taken meanwhile: try the next one
                print(f"vast: offer {offer['id']} refused: {(e.stderr or e.stdout or '').strip()[:200]}", flush=True)
                continue
            iid = made["new_contract"]
            print(f"vast: rented instance {iid} (offer {offer['id']}, ${offer['dph_total']:.3f}/h, "
                  f"{offer.get('geolocation')})", flush=True)
            target = self.wait(iid)
            if target:
                return iid, target
            self.destroy(iid)
        raise RuntimeError("could not rent a reachable RTX 5090 VM")

    def wait(self, iid):
        deadline = self.clock() + self.boot_timeout
        while self.clock() < deadline:
            inst = next((i for i in self.instances() if i["id"] == iid), None)
            if inst is None:
                return None
            t = ssh_target(inst)
            if inst.get("actual_status") == "running" and t and self.ssh_ok(iid, t):
                return t
            self.sleep(15)
        print(f"vast: instance {iid} not reachable after {self.boot_timeout} s", flush=True)
        return None

    def used(self):
        self.last_used = self.clock()

    def release_if_idle(self):
        if self.clock() - self.last_used < self.idle:
            return
        for inst in self.instances():
            self.destroy(inst["id"])
