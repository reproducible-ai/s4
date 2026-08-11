"""Diagnostic ONLY — not part of the reproduction, never traced.

Imports exactly the module graph that upstream `train.py` pulls in, initialises
CUDA, then lists every native shared object the process has mapped and whether a
dpkg package owns it. Run from the repo root in the (untraced) setup stage:

    python repro/native_libs.py

Rationale: roar records dpkg-owned system libraries and silently drops any `.so`
that no package owns, so "what loaded but was NOT recorded" is invisible in the
DAG and has to be measured on the host while it is alive.
"""

import subprocess
import sys


def main() -> int:
    import torch  # noqa: F401
    import hydra  # noqa: F401
    import pytorch_lightning  # noqa: F401
    import wandb  # noqa: F401
    import numpy  # noqa: F401

    import src.models.nn.utils  # noqa: F401
    import src.utils  # noqa: F401
    import src.utils.train  # noqa: F401
    from src.dataloaders import SequenceDataset  # noqa: F401
    from src.tasks import decoders, encoders, tasks  # noqa: F401

    print("python     ", sys.version.split()[0])
    print("torch      ", torch.__version__)
    print("cuda avail ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device     ", torch.cuda.get_device_name(0))
        x = torch.zeros(1024, device="cuda")
        # touch cuFFT + cuBLAS so their .so files are mapped, as S4 does
        torch.fft.rfft(x)
        (x.view(32, 32) @ x.view(32, 32)).sum().item()
        torch.cuda.synchronize()

    paths = set()
    with open("/proc/self/maps") as fh:
        for line in fh:
            p = line.rstrip("\n").split(" ")[-1]
            if p.startswith("/") and (".so" in p):
                paths.add(p)

    owned, unowned = [], []
    for p in sorted(paths):
        r = subprocess.run(["dpkg", "-S", p], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            owned.append((p, r.stdout.strip().split(":")[0]))
        else:
            unowned.append(p)

    print(f"\n=== NATIVE LIBS: {len(owned)} dpkg-owned / {len(unowned)} unowned ===")
    for p, pkg in owned:
        print(f"DPKG-OWNED  {pkg:<28} {p}")
    for p in unowned:
        print(f"NO-DPKG     {'-':<28} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
