"""Collapse duplicate installed distributions before capture — NOT part of the model.

Why this exists, precisely:

`train.py` imports pytorch_lightning 2.0.4, whose `lightning_fabric/__init__.py`
calls `__import__("pkg_resources").declare_namespace(...)`. Importing
`pkg_resources` builds a working set by reading the METADATA of **every**
distribution installed on the host. A provenance tracer that attributes files to
packages therefore sees the host's *entire* site-packages inventory, not this
recipe's import closure — so anything broken in that inventory lands in the
recorded freeze.

This AMI ships `importlib-metadata` twice, under both PEP-503 spellings and at two
different versions (`importlib-metadata 6.11.0` and `importlib_metadata 8.0.0`).
Both spellings normalise to the same project, so a freeze containing both is
unsatisfiable: pip/uv answer `ResolutionImpossible`, and the row can never rebuild.

The remedy is to delete the stale `*.dist-info` directory (keeping the newest
version), which is what a clean upgrade would have done. Nothing else is touched:
only `.dist-info` / `.egg-info` directories under a `site-packages` /
`dist-packages` tree are ever removed, and only when two of them normalise to the
same project name.

    python -m repro.fix_dup_dists          # report + fix
    python -m repro.fix_dup_dists --dry-run
"""

from __future__ import annotations

import importlib.metadata as im
import os
import re
import shutil
import sys


def canon(name: str) -> str:
    """PEP 503 normalisation."""
    return re.sub(r"[-_.]+", "-", (name or "")).lower()


def version_key(v: str):
    try:
        from packaging.version import Version

        return (1, Version(v))
    except Exception:
        return (0, tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[^0-9]+", v or "")))


def dist_dir(d) -> str | None:
    p = getattr(d, "_path", None)
    if p is None:
        return None
    p = str(p)
    return p if p.endswith((".dist-info", ".egg-info")) else None


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv

    groups: dict[str, list] = {}
    for d in im.distributions():
        try:
            name = d.metadata["Name"]
        except Exception:
            continue
        if not name:
            continue
        groups.setdefault(canon(name), []).append(d)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"installed distributions: {sum(len(v) for v in groups.values())}"
          f" · duplicate project names: {len(dupes)}")
    if not dupes:
        print("no PEP-503 duplicates — nothing to do")
        return 0

    removed = 0
    for name, ds in sorted(dupes.items()):
        ds = sorted(ds, key=lambda d: version_key(d.version), reverse=True)
        keep = ds[0]
        print(f"\nDUPLICATE {name}:")
        for d in ds:
            print(f"    {d.metadata['Name']}=={d.version}  {dist_dir(d)}"
                  f"{'   <-- keep' if d is keep else ''}")
        for d in ds[1:]:
            p = dist_dir(d)
            if not p:
                print(f"    skip (no dist-info path): {d.metadata['Name']}=={d.version}")
                continue
            if not any(seg in p for seg in ("site-packages", "dist-packages")):
                print(f"    skip (not under site-packages): {p}")
                continue
            if not os.path.isdir(p):
                print(f"    skip (not a directory): {p}")
                continue
            if dry:
                print(f"    would remove {p}")
            else:
                shutil.rmtree(p)
                print(f"    removed {p}")
                removed += 1

    print(f"\nremoved {removed} stale dist-info director{'y' if removed == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
