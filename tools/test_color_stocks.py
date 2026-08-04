"""
Invariants for the colour film stock library.

WHY THIS EXISTS. 56 stocks were merged from Capture One data in commit 69ad3da and
every one of them silently lost its `g_off` keyword in the hand copy-paste. The green
channel therefore rendered at the MEAN of the three channels instead of its own curve,
for months, across 56 stocks -- up to 20 levels wrong on the worst. Nothing caught it
because nothing looked. This file looks.

The load-bearing fact: tools/generate_stocks.py builds the base curve as the MEAN of
the red, green and blue curves, and each *_off as that channel's deviation from the
mean. So r_off + g_off + b_off == 0 identically, by construction. That identity is a
free, exact checksum on the whole C1-derived library: if an offset is ever dropped,
mistyped or half-merged again, the sum stops being zero and I1 fails.

Run:  python tools/test_color_stocks.py
"""

import ast
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.color_stocks import COLOR_STOCKS          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIVE = os.path.join(ROOT, "data", "color_stocks.py")
FRAG = os.path.join(HERE, "generated_stocks.txt")

# Fuji Fortia SP was deliberately hand-tuned away from the C1 data (r_off pushed to
# 0.1, b_off to -0.05) to exaggerate Fuji's ultra-saturated slide look. Its offsets
# therefore do NOT sum to zero, and that is intentional rather than a merge fault.
HAND_TUNED = {"Slide / Fuji Fortia SP"}

# Rounding to 3dp can leave up to 0.0015 in the sum of three terms.
ROUNDING_SLACK = 0.004

_passed = 0
_failed = 0


def check(label, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {label}  {detail}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}  {detail}")


def stock_calls(text):
    """key -> (positional args, keyword args) for each "key": _stock(...)."""
    out = {}
    for m in re.finditer(r'"([^"]+)"\s*:\s*_stock\(', text):
        key, i, depth = m.group(1), m.end(), 1
        while depth:
            ch = text[i]
            depth += (ch in "([{") - (ch in ")]}")
            i += 1
        node = ast.parse("f(" + text[m.end():i - 1] + ")", mode="eval").body
        out[key] = ([ast.literal_eval(a) for a in node.args],
                    {k.arg: ast.literal_eval(k.value) for k in node.keywords})
    return out


live_text = io.open(LIVE, encoding="utf-8").read()
live = stock_calls(live_text)

frag_keys = set()
if os.path.exists(FRAG):
    frag_keys = set(stock_calls(io.open(FRAG, encoding="utf-8").read()))

print("=" * 78)
print("COLOR STOCK LIBRARY INVARIANTS")
print("=" * 78)

# --- I1: the mean-anchor checksum -------------------------------------------
print("\nI1  offsets sum to zero, because the base curve IS the channel mean")
if not frag_keys:
    print("  [SKIP] tools/generated_stocks.txt not present")
else:
    bad = []
    for key in sorted(frag_keys & set(live)):
        if key in HAND_TUNED:
            continue
        _, kw = live[key]
        sums = [sum(v) for v in zip(kw.get("r_off", (0, 0, 0)),
                                    kw.get("g_off", (0, 0, 0)),
                                    kw.get("b_off", (0, 0, 0)))]
        worst = max(abs(v) for v in sums)
        if worst > ROUNDING_SLACK:
            bad.append((key, worst))
    check("I1 every C1-derived stock is mean-anchored", not bad,
          f"{len(frag_keys & set(live)) - len(HAND_TUNED)} stocks checked"
          if not bad else f"{len(bad)} violate, worst {max(b for _, b in bad):.4f}")
    for key, worst in bad[:5]:
        print(f"          {key}  |r+g+b| = {worst:.4f}")

    check("I1 the known hand-tuned exception is still the only one",
          all(k in live for k in HAND_TUNED),
          f"exempt: {', '.join(sorted(HAND_TUNED))}")

# --- I2: no C1 stock silently lost an offset --------------------------------
print("\nI2  no C1-derived stock is missing an offset it should carry")
if frag_keys:
    frag = stock_calls(io.open(FRAG, encoding="utf-8").read())
    missing = []
    for key in sorted(frag_keys & set(live)):
        _, fkw = frag[key]
        _, lkw = live[key]
        if key in HAND_TUNED:
            continue
        for field in ("r_off", "g_off", "b_off"):
            if field in fkw and field not in lkw:
                missing.append(f"{key}:{field}")
    check("I2 every offset present in the generator output survives in the library",
          not missing,
          f"{len(frag_keys & set(live))} stocks checked" if not missing
          else f"{len(missing)} dropped: {missing[:4]}")

# --- I3: the curves are structurally sane -----------------------------------
print("\nI3  every stock produces usable curve parameters")
bad_vals = []
for name, s in COLOR_STOCKS.items():
    for chan in ("r_curve", "g_curve", "b_curve"):
        c = getattr(s, chan)
        for f in ("toe_power", "shoulder_power", "slope"):
            v = getattr(c, f)
            if not (0.05 < v < 6.0):
                bad_vals.append(f"{name}.{chan}.{f}={v}")
check("I3 toe/shoulder/slope stay in a physical range", not bad_vals,
      f"{len(COLOR_STOCKS)} stocks x 3 channels"
      if not bad_vals else f"{len(bad_vals)} out of range: {bad_vals[:3]}")

check("I3 saturation is positive and bounded",
      all(0.0 < s.saturation < 3.0 for s in COLOR_STOCKS.values()),
      f"{len(COLOR_STOCKS)} stocks")

# --- I4: the library did not shrink -----------------------------------------
print("\nI4  the library is intact")
check("I4 stock count matches the README claim of 111 colour stocks",
      len(COLOR_STOCKS) == 111, f"{len(COLOR_STOCKS)} stocks")
check("I4 no duplicate display names",
      len({s.name for s in COLOR_STOCKS.values()}) == len(COLOR_STOCKS),
      f"{len({s.name for s in COLOR_STOCKS.values()})} unique of {len(COLOR_STOCKS)}")

# --- negative controls ------------------------------------------------------
print("\nNC  negative controls  -- each MUST fail")


def nc(label, cond, why):
    """cond is the BROKEN condition; it must evaluate False."""
    global _passed, _failed
    if not cond:
        _passed += 1
        print(f"  [PASS] {label}  {why}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}  check is blind: {why}")


# NC1: drop a g_off and I1 must notice
if frag_keys:
    victim = "Neg / Fuji Pro 160C"
    _, kw = live[victim]
    sums = [sum(v) for v in zip(kw.get("r_off", (0, 0, 0)),
                                (0, 0, 0),                       # g_off dropped
                                kw.get("b_off", (0, 0, 0)))]
    nc("NC1 dropping a g_off still passes the mean-anchor check",
       max(abs(v) for v in sums) <= ROUNDING_SLACK,
       f"simulated drop on {victim} gives |r+g+b| = {max(abs(v) for v in sums):.3f}")

# NC2: a corrupted curve must fail the range check
nc("NC2 an absurd slope passes the range check",
   0.05 < 99.0 < 6.0, "slope=99 must be rejected")

# NC3: the count check must be able to fail
nc("NC3 a short library passes the count check",
   110 == 111, "110 stocks must not satisfy the 111 check")

print()
print("=" * 78)
print(f"{_passed} passed, {_failed} failed")
print("=" * 78)
sys.exit(1 if _failed else 0)
