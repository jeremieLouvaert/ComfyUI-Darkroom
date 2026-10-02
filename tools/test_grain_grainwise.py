"""
Grain-wise Film Grain Pro engine == pixel-wise engine, BITWISE
(docs/film-grain-pro-grainwise.md).

Runs both _render_channel implementations on the same inputs across grain
size, radius variation, sample count, seed, image shape (incl. non-square and
borders that wrap) and checks torch.equal. Negative control: a 0.1% change in
grain radius must break equality.

Run: python.exe tools/test_grain_grainwise.py
"""

import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gn", os.path.join(os.path.dirname(HERE), "utils", "grain_newson.py"))
gn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gn)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")
    return ok


def both(u, mu, sig, n, fs, seed):
    a = gn._render_channel(u, mu, sig, n, fs, seed, DEV)
    b = gn._render_channel_grainwise(u, mu, sig, n, fs, seed, DEV, max_pairs=2_000_000)  # force chunking
    return a, b


if __name__ == "__main__":
    g = torch.Generator().manual_seed(5)
    shapes = [(64, 96), (97, 53), (128, 128), (40, 300)]
    cases = 0
    for (H, W) in shapes:
        u = torch.rand(H, W, generator=g)
        u[:4] = 0.0                       # black rows (no grains)
        u[-4:] = 1.0                      # white rows (dense)
        for mu in (0.5, 1.3, 3.2):
            for sig in (0.0, 0.4 * mu):
                for n in (4, 16):
                    for seed in (0, 17):
                        a, b = both(u, mu, sig, n, 0.8, seed)
                        cases += 1
                        check(f"{H}x{W} mu={mu} sig={sig:.2f} N={n} seed={seed}", torch.equal(a, b),
                              f"max diff {(a - b).abs().max().item():.3e}")
    # Jeremie's settings scale: mu ~5.7 px, N=64, on a 256x400 crop
    u = torch.rand(256, 400, generator=g)
    a, b = both(u, 5.7, 0.0, 64, 0.8, 0)
    cases += 1
    check("mu=5.7 N=64 (large grain)", torch.equal(a, b), f"max diff {(a - b).abs().max().item():.3e}")
    print(f"{cases} equality cases run")

    # negative control: 0.1% radius change must be detected
    a = gn._render_channel(u, 5.7, 0.0, 16, 0.8, 0, DEV)
    b = gn._render_channel_grainwise(u, 5.7 * 1.001, 0.0, 16, 0.8, 0, DEV)
    check("negative control (radius x1.001) is detected", not torch.equal(a, b))

    # timing, same crop
    for name, f in (("pixel-wise", gn._render_channel), ("grain-wise", gn._render_channel_grainwise)):
        torch.cuda.synchronize() if DEV == "cuda" else None
        t = time.perf_counter()
        f(u, 5.7, 0.0, 64, 0.8, 0, DEV)
        torch.cuda.synchronize() if DEV == "cuda" else None
        print(f"  {name}: {(time.perf_counter() - t) * 1000:.0f} ms on 256x400, mu 5.7, N 64")
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
