// ComfyUI-Darkroom -- grading preset ghost overlay, shared by wheels and curves.
//
// The grading nodes apply their presets in PYTHON at execute time, so with a
// preset active the widget values are the MANUAL offset only and the canvas has
// never seen the preset's numbers. That is why these nodes used to carry the
// caption "preset active, curve shows the manual offset only".
//
// This module fetches the preset tables once from /darkroom/grading_presets and
// resolves `effective = manual (+) preset` locally, so the ghost tracks a drag
// with no server round trip. Python stays the single source of both the numbers
// AND the resolution rules -- nothing here hardcodes a preset value or a widget
// name.
//
// The rules are not all additive, which is the whole reason they are served
// rather than assumed:
//   add           effective = manual + v
//   set           effective = v
//   set_if_below  effective = v  only while manual[gate] < thr, else manual
//                 (the wheels keep the MANUAL hue once that zone's saturation
//                  or intensity is turned up -- log_wheels.py:109)
// and `bypass`, where the preset disables the node outright (Log Wheels'
// "Neutral - reset all" returns the input image and discards the manual values),
// so the honest ghost is a neutral one, not the manual grade.
//
// Pinned by tools/test_preset_ops.py: 220 preset x manual-value combinations
// compared against the real nodes, bitwise, with 7 negative controls.

const ROUTE = "/darkroom/grading_presets";

let _data = null;
let _pending = null;
let _failed = false;

// Fetched once per page. A failure is logged and then treated as "no preset
// data", which degrades to exactly the pre-ghost behaviour rather than breaking
// the widget.
export function loadPresets(onReady) {
  if (_data || _failed) return _data;
  if (!_pending) {
    // fetch() can throw SYNCHRONOUSLY on a relative URL outside a browser
    // (headless module teeth, any non-document host), which a bare .catch()
    // would never see. Both paths degrade to "no preset data".
    try {
      _pending = fetch(ROUTE)
        .then((r) => {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then((d) => { _data = d; return d; })
        .catch((e) => {
          _failed = true;
          console.error("[Darkroom] preset ghost unavailable (" + ROUTE + "): " +
                      ((e && e.message) || e));
          return null;
        });
    } catch (e) {
      _failed = true;
      console.error("[Darkroom] preset ghost unavailable (" + ROUTE + "): " +
                      ((e && e.message) || e));
      _pending = Promise.resolve(null);
    }
  }
  if (onReady) _pending.then(onReady);
  return _data;
}

export function presetsReady() { return !!_data; }

export function customName() {
  return (_data && _data.custom) || "Custom (manual)";
}

// The preset entry for a node's current selection, or null when there is
// nothing to ghost (no data yet, unknown node, or "Custom (manual)").
export function presetEntry(nodeType, presetName) {
  if (!_data || !nodeType) return null;
  const n = _data.nodes[nodeType];
  if (!n) return null;
  if (!presetName || presetName === _data.custom) return null;
  return n.presets[presetName] || null;
}

// readVal(widgetName) -> number. Returns a plain object of effective values for
// the widgets asked for. Gates read the MANUAL values, never the running
// result, matching the nodes -- they test the incoming argument.
export function effectiveValues(readVal, entry, widgets) {
  const out = {};
  for (const w of widgets) out[w] = readVal(w);
  if (!entry) return out;
  if (entry.bypass) {
    for (const w of widgets) out[w] = 0;
    return out;
  }
  for (const o of entry.ops || []) {
    if (!(o.w in out)) continue;
    if (o.op === "add") out[o.w] = readVal(o.w) + o.v;
    else if (o.op === "set") out[o.w] = o.v;
    else if (o.op === "set_if_below" && readVal(o.gate) < o.thr) out[o.w] = o.v;
  }
  return out;
}

// True when the effective values differ enough from the manual ones to be worth
// drawing a second time. Below this the ghost would sit under the solid mark and
// read as a rendering artefact.
export function differs(manual, effective, eps) {
  const e = eps === undefined ? 0.5 : eps;
  for (const k in effective) {
    if (Math.abs((effective[k] || 0) - (manual[k] || 0)) >= e) return true;
  }
  return false;
}

export const GHOST_STROKE = "rgba(201,162,39,0.85)";
export const GHOST_FILL = "rgba(201,162,39,0.16)";
export const GHOST_DASH = [3, 3];
