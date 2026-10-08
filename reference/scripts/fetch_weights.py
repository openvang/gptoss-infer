"""Download the pinned gpt-oss-20b checkpoint and verify every file against reference/weights.lock.json.

    python reference/scripts/fetch_weights.py --out /path/to/models/gpt-oss-20b
    python reference/scripts/fetch_weights.py --out /path/to/models/gpt-oss-20b --verify-only

LFS files (weights, tokenizer) are checked by sha256, small files by their git blob sha1, so every byte is
pinned to the lock's commit. Downloads resume; a file that fails verification is deleted, never used.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "weights.lock.json"


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(path, entry):
    """Return None if `path` matches `entry`, else a reason string."""
    if not path.exists():
        return "missing"
    if entry.get("size") is not None and path.stat().st_size != entry["size"]:
        return f"size {path.stat().st_size} != {entry['size']}"
    if entry.get("sha256"):
        got = sha256_file(path)
        return None if got == entry["sha256"] else f"sha256 {got} != {entry['sha256']}"
    if entry.get("git_blob_sha1"):
        got = git_blob_sha1(path)
        return None if got == entry["git_blob_sha1"] else f"git blob {got} != {entry['git_blob_sha1']}"
    return "lock entry has no hash"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory to hold the checkpoint")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    lock = json.loads(LOCK.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        from huggingface_hub import hf_hub_download

        # Small files first, so a config problem shows up before 13 GB of downloads.
        for name in sorted(lock["files"], key=lambda n: lock["files"][n]["size"] or 0):
            if verify(out / name, lock["files"][name]) is None:
                print(f"ok (cached)  {name}", flush=True)
                continue
            print(f"downloading  {name}", flush=True)
            hf_hub_download(repo_id=lock["repo"], filename=name, revision=lock["revision"], local_dir=out)

    bad = {}
    for name, entry in sorted(lock["files"].items()):
        reason = verify(out / name, entry)
        print(f"{'VERIFIED' if reason is None else 'FAILED  '}  {name}" + (f"  ({reason})" if reason else ""), flush=True)
        if reason:
            bad[name] = reason
    if bad:
        for name in bad:
            if (out / name).exists() and not args.verify_only:
                os.remove(out / name)
        sys.exit(f"{len(bad)} file(s) failed verification; failed downloads were deleted")
    (out / "VERIFIED.json").write_text(json.dumps({"repo": lock["repo"], "revision": lock["revision"]}, indent=2))
    print(f"all {len(lock['files'])} files verified against {lock['repo']}@{lock['revision'][:12]}")


if __name__ == "__main__":
    main()
