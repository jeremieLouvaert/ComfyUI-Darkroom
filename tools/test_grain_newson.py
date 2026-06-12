"""
Verification + performance gate for the Newson Film Grain Pro engine.
Run with the ComfyUI embedded python. Writes a report to test_grain_newson_report.txt.

Teeth: tone-preservation, midtone-peaked grain amplitude (bell), resolution
independence, determinism, and a hard perf gate on 1024x1024.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/utils")
import torch
import grain_newson as gn

REPORT = []
def log(s):
    print(s)
    REPORT.append(str(s))

def flat(val, h, w, c=3):
    return torch.full((h, w, c), float(val), dtype=torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device = {device}, torch = {torch.__version__}")

PASS = True
def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    log(f"  [{status}] {name}  {detail}")

# ---------------------------------------------------------------------------
# 1. Finite, in-range, grain present
# ---------------------------------------------------------------------------
log("\n[1] basic sanity (256x256, flat 0.5)")
out = gn.render_film_grain(flat(0.5, 256, 256), grain_size=1.2, radius_variation=0.0,
                           strength=1.0, color_grain=0.0, n_samples=64,
                           filter_sigma=0.8, seed=7, device=device)
out = out.cpu()
check("finite", torch.isfinite(out).all().item())
check("in [0,1]", (out.min().item() >= 0.0) and (out.max().item() <= 1.0),
      f"min={out.min():.3f} max={out.max():.3f}")
std = out.std().item()
check("grain present (std>0.01)", std > 0.01, f"std={std:.4f}")

# ---------------------------------------------------------------------------
# 2. Tone preservation across gray levels
# ---------------------------------------------------------------------------
log("\n[2] tone preservation (mean out ~= input)")
for val in (0.2, 0.5, 0.8):
    o = gn.render_film_grain(flat(val, 256, 256), 1.2, 0.0, 1.0, 0.0, 96, 0.8, 11, device).cpu()
    m = o.mean().item()
    check(f"gray {val}: mean={m:.4f}", abs(m - val) < 0.02, f"drift={m-val:+.4f}")

# ---------------------------------------------------------------------------
# 3. Grain amplitude is midtone-peaked (teeth: white-noise would be flat)
# ---------------------------------------------------------------------------
log("\n[3] grain amplitude bell (std mid > std shadow and > std highlight)")
def grain_std(val):
    o = gn.render_film_grain(flat(val, 256, 256), 1.2, 0.0, 1.0, 0.0, 96, 0.8, 5, device).cpu()
    return (o - val).std().item()
s_lo, s_mid, s_hi = grain_std(0.06), grain_std(0.5), grain_std(0.94)
log(f"  std: shadow(0.06)={s_lo:.4f}  mid(0.5)={s_mid:.4f}  highlight(0.94)={s_hi:.4f}")
check("mid > shadow", s_mid > s_lo)
check("mid > highlight", s_mid > s_hi)

# ---------------------------------------------------------------------------
# 4. Resolution independence (grain std comparable at 256 vs 512, same grain_size)
# ---------------------------------------------------------------------------
log("\n[4] resolution independence (per-pixel grain std stable across res)")
o256 = gn.render_film_grain(flat(0.5, 256, 256), 1.2, 0.0, 1.0, 0.0, 96, 0.8, 3, device).cpu()
o512 = gn.render_film_grain(flat(0.5, 512, 512), 1.2, 0.0, 1.0, 0.0, 96, 0.8, 3, device).cpu()
g256, g512 = (o256 - 0.5).std().item(), (o512 - 0.5).std().item()
ratio = max(g256, g512) / max(min(g256, g512), 1e-6)
log(f"  std@256={g256:.4f}  std@512={g512:.4f}  ratio={ratio:.2f}")
check("std ratio < 1.6", ratio < 1.6)

# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------
log("\n[5] determinism")
a = gn.render_film_grain(flat(0.5, 128, 128), 1.2, 0.0, 1.0, 0.0, 32, 0.8, 42, device).cpu()
b = gn.render_film_grain(flat(0.5, 128, 128), 1.2, 0.0, 1.0, 0.0, 32, 0.8, 42, device).cpu()
c = gn.render_film_grain(flat(0.5, 128, 128), 1.2, 0.0, 1.0, 0.0, 32, 0.8, 43, device).cpu()
check("same seed -> identical", torch.allclose(a, b))
check("diff seed -> differs", not torch.allclose(a, c))

# ---------------------------------------------------------------------------
# 6. PERF GATE (1024x1024)
# ---------------------------------------------------------------------------
log("\n[6] PERF GATE (1024x1024, N=64)")
img = (torch.rand(1024, 1024, 3) * 0.6 + 0.2)
# warmup
_ = gn.render_film_grain(flat(0.5, 64, 64), 1.2, 0.0, 1.0, 0.0, 8, 0.8, 1, device)
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
_ = gn.render_film_grain(img, 1.2, 0.0, 0.5, 0.0, 64, 0.8, 9, device)
if device == "cuda":
    torch.cuda.synchronize()
t_luma = time.time() - t0
t0 = time.time()
_ = gn.render_film_grain(img, 1.2, 0.0, 0.5, 1.0, 64, 0.8, 9, device)
if device == "cuda":
    torch.cuda.synchronize()
t_color = time.time() - t0
log(f"  luma(cg=0): {t_luma:.2f}s   color(cg=1): {t_color:.2f}s")
check("luma 1024 under 5s", t_luma < 5.0, f"{t_luma:.2f}s")
log(f"  (color is ~3x luma by design; gate is on the default luma path)")

# variable radius smoke test
log("\n[7] variable radius smoke (radius_variation=0.4)")
ov = gn.render_film_grain(flat(0.5, 256, 256), 1.2, 0.4, 1.0, 0.0, 64, 0.8, 2, device).cpu()
check("var-radius finite & in range",
      torch.isfinite(ov).all().item() and ov.min() >= 0 and ov.max() <= 1)

log("\n" + ("ALL PASS" if PASS else "SOME FAILED"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "test_grain_newson_report.txt"), "w") as f:
    f.write("\n".join(REPORT))
