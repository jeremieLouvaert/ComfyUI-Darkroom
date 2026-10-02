# Film Grain Pro: grain-wise evaluation (same output, less work)

## Problem
On an 8336 x 2688 image, Film Grain Pro takes 495 s (grain_size 0.7, N = 64).

`utils/grain_newson.py::_render_channel` is pixel-wise with cell size 1 px:
- for every Monte-Carlo sample it visits all (2R+1)^2 neighbour cells, where R = ceil(rm);
- it rolls three (H, W, qmax) tensors per cell.

With grain_size relative to a 1024 px reference, mu_r = 0.7 x 8336/1024 = 5.7 px, so R = 6:
- 169 cells x 64 samples = 10,816 full-frame passes;
- a cell holds lambda = -ln(1-u)/(pi r^2), about 0.007 grains on average, so roughly 99% of that work tests empty cells.

## What the current code computes, exactly
For sample s with offset xi = (xix, xiy):
- `sx = floor(0.5 + xix)`, `fracx = 0.5 + xix - sx` (Python floats), and the same for y.

For pixel (i, j) and neighbour offset (dcx, dcy) in [-R, R]^2:
- the cell is `cx = (j + sx + dcx) mod W` and `cy = (i + sy + dcy) mod H`. The modulo comes from `torch.roll`, so the field is toroidal.
- For each grain slot k < q[cy, cx], with sub-cell position (ux, uy) and squared radius r2 (rm^2, or r_grid^2 when sigma_r > 0):

      dx = (fracx - dcx) - ux        # (python float) - float32 tensor
      dy = (fracy - dcy) - uy
      hit = dx*dx + dy*dy < r2

- `covered[i, j] = OR over all (dcx, dcy, k) of hit`, and `v += covered` per sample in sample order.

## Grain-wise, same arithmetic
Invert the cell lookup. A grain in cell (cx, cy) can only be hit from pixel `j = (cx - sx - dcx) mod W` and `i = (cy - sy - dcy) mod H`, for the same (dcx, dcy) in [-R, R]^2.

So for every existing grain (cell, k) and every (dcx, dcy):
- evaluate exactly the expression above, with the same operand types and order;
- scatter the hits into `covered` (a boolean OR, which is order-independent).

It is the same set of (pixel, cell, k) tests restricted to cells where k < q[cell]. The tests skipped are exactly those whose `present` term was False. Therefore `covered`, and hence the output, is **bitwise identical**. No change to the model, the realisation (hash field), the parameters, or the tone.

Work per sample: pixel-wise is about H x W x qmax x (2R+1)^2 tests; grain-wise is about (number of grains) x (2R+1)^2. The ratio is qmax / lambda, about 2 / 0.007, roughly 300x fewer tests here. The ratio shrinks for small grains or bright images, where lambda is larger. Grain-wise is never worse than about 1x because #grains <= H x W x qmax.

## Acceptance (spike)
- **Bitwise equality** (`torch.equal`) of `_render_channel` vs the grain-wise version on random images across:
  - grain_size in {0.3, 0.7, 1.5}, radius_variation in {0, 0.4}, N in {4, 16, 64};
  - several seeds and sizes, including non-square and wrap-around borders;
  - luminance and colour modes.
- **A negative control:** perturbing the hit radius by 1e-3 must break equality.
- **Timing** on the full 8336 x 2688 case with Jeremie's settings, against the 495 s baseline.
- **Memory:** pair tensors are chunked so peak VRAM stays bounded.
