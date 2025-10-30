
# Orbbec Viewer — Quick User Guide

**RGB top | Depth/Height bottom**  
Keyboard-only viewer with live overlays and a simple probe. Calibration uses 4 ArUco markers.

---

## What it does
- Streams RGB (top) and DEPTH or HEIGHT (bottom) panels.
- HEIGHT is computed from a fitted ground plane after calibration.
- Publishes frames + metadata over **ZeroMQ PUB** for other apps to subscribe to.
- No mouse tools: this app is intentionally minimal.

**Overlays**
- **Top‑left:** current mode + context instructions **and the active mode’s numeric range** in mm.
- **Top‑right:** when the probe is enabled, shows a single numeric readout (`Depth: N mm` or `Height: N mm`).
- **Center (temporary):** error banner when calibration is requested but not all markers are visible — *"Need 4 visible markers. IDs 1,2,3,4"*.

---

## Controls
| Key | Action |
|---|---|
| **T** | Toggle **HEIGHT ↔ DEPTH** |
| **P** | Toggle **probe** (green crosshair) |
| **W / A / S / D** | Move probe (accelerates while holding) |
| **Arrow keys** | Move probe (accelerates while holding) |
| **C** | Start (re)calibration when **ArUco IDs 1,2,3,4** are visible |
| **Q** | Quit |

**Probe acceleration**
- Reaches **max speed in ~2 seconds by default**.  
- Hold a direction to ramp up; quick taps do discrete steps.  
- If you pause longer than `MID_ACCEL_RELEASE_TIMEOUT_S` (default **0.25s**), the hold resets.

---

## Requirements
- **Python 3.8+**
- **Orbbec SDK** runtime and Python bindings (`pyorbbecsdk`) installed and your Orbbec camera connected.
- OpenCV **with `aruco`** module: `opencv-contrib-python`
- `numpy`, `pyzmq`

```bash
# Typical packages (adjust per platform/virtualenv)
pip install opencv-contrib-python numpy pyzmq
# Install Orbbec Python SDK per vendor docs (package name/steps can vary)
# For Linux you may also need: libgl1, v4l2, udev rules for the camera
```

> If `import pyorbbecsdk` fails, install the Orbbec SDK / Python wheel from Orbbec’s official site and confirm your Python version matches the wheel.

---

## Running
```bash
python orbbec_viewer.py
```
A resizable window titled **“Orbbec Viewer (RGB ↑ | Depth/Height ↓)”** will open.

---

## Environment variables (tunables)

| Name | Default | Meaning |
|---|---:|---|
| `MID_PUB` | `tcp://127.0.0.1:5555` | ZeroMQ PUB bind endpoint |
| `MID_CAL_FRAMES` | `30` (min 30) | Frames to accumulate during calibration |
| `MID_LOG_LEVEL` | `INFO` | Python logging level |
| `MID_DEPTH_VIZ_MIN_M` | `0.20` | Depth colormap min (meters) |
| `MID_DEPTH_VIZ_MAX_M` | `1.00` | Depth colormap max (meters) |
| `MID_HEIGHT_VIZ_MIN_M` | `0.001` | Height colormap min (meters) |
| `MID_HEIGHT_VIZ_MAX_M` | `0.20` | Height colormap max (meters) |
| `MID_BLOB_ENABLE` | `1` | Hole-filling on (0 = off) |
| `MID_BLOB_TOL_MM` | `10` | Hole fill tolerance (mm) |
| `MID_BLOB_RING_PX` | `3` | Border-ring radius (px) for fill consistency |
| `MID_BLOB_MIN_BORDER` | `24` | Min ring pixels for a hole to be filled |
| `MID_PROBE_STEP` | `3` | Base step in pixels for the probe |
| `MID_ACCEL_MAX_MULT` | `8.0` | Max acceleration multiplier for probe speed |
| `MID_ACCEL_TIME_TO_MAX_S` | `2.0` | Time to reach max speed (s); used to derive gain if `MID_ACCEL_GAIN` is not set |
| `MID_ACCEL_GAIN` | *(derived)* | Accel gain per second. If set, overrides derivation |
| `MID_ACCEL_REPEAT_GRACE_S` | `0.12` | Max delay between repeat presses to keep a “hold” |
| `MID_ACCEL_RELEASE_TIMEOUT_S` | `0.25` | Hold stops after this idle time (s) |
| `MID_CAL_MIN_VALID` | `500` | Min valid ROI points for plane fit |
| `MID_CAL_FILE` | `calibration_plane.json` | Where calibration is persisted |

**Note on acceleration:** if `MID_ACCEL_GAIN` is unset, the app derives it so that  
`1 + gain * MID_ACCEL_TIME_TO_MAX_S = MID_ACCEL_MAX_MULT`.

---

## Calibration (HEIGHT mode)
1. Print or display **ArUco DICT_4X4_50** markers **IDs 1, 2, 3, 4**.
2. Place them in view of the camera; the viewer outlines detected markers on the RGB panel.
3. Press **C**. If any marker is missing, a **centered red banner** appears:
   - *“Need 4 visible markers. IDs 1,2,3,4”*
4. When all are visible, the app captures `MID_CAL_FRAMES` frames, fits a plane inside the convex hull of markers, and saves:
   - **`calibration_plane.json`** (plane normal & offset, intrinsics if available, ROI hull, viz ranges).
5. On success, **HEIGHT** mode activates and becomes the default on next launch.

**Resetting calibration**
- Delete **`calibration_plane.json`** to force recalibration next run.

---

## Panels & Ranges
- **DEPTH panel (bottom)**: JET colormap of depth (mm) **clipped** to `[MID_DEPTH_VIZ_MIN_M, MID_DEPTH_VIZ_MAX_M]`. Values outside the range or inside ArUco masks render as 0 (no color).
- **HEIGHT panel (bottom)**: JET colormap of height above the fitted plane (mm) **clipped** to `[MID_HEIGHT_VIZ_MIN_M, MID_HEIGHT_VIZ_MAX_M]`.
- The **top-left instruction block** always shows the **active panel’s range in mm** (e.g., `Range: 200–1000 mm`).

---

## Probe
- Toggle with **P**. A **green crosshair** appears on both panels.
- Move with **W/A/S/D** or **arrow keys**. Holding a direction accelerates up to `MID_ACCEL_MAX_MULT × MID_PROBE_STEP` in ~`MID_ACCEL_TIME_TO_MAX_S` seconds.
- The **top-right** shows a single value: `Depth: N mm` in DEPTH mode or `Height: N mm` in HEIGHT mode.

---

## ZeroMQ PUB output
The app publishes one JSON message per frame on `MID_PUB`.

**Common fields (abbrev.)**
```jsonc
{
  "type": "frame",
  "timestamp": "2025-01-01T12:34:56.789Z",
  "frame_name": "20250101_123456_789",
  "aligned_to_color": true,
  "rgb_jpeg_b64": "…",
  "rgb_annotated_jpeg_b64": "…",
  "visualization": {
    "enabled": true,
    "heatmap_jpeg_b64": "…"
  },
  "aruco": { "ids": [1,2,3,4], "masked_pixels": 1234 },
  "meta": {
    "has_plane": true,
    "selected_mode": "height",
    "effective_mode": "height",
    "viz_ranges_m": { "depth": {"min":0.2,"max":1.0}, "height":{"min":0.001,"max":0.2} },
    "depth_scale_mm_per_unit": 1.0,
    "blob_fill": {"enabled": true, "tol_mm": 10, "ring_px": 3, "min_border": 24},
    "acceleration": {"max_mult": 8.0, "gain_per_s": 3.5}
  },
  // If effective_mode == "height":
  "depth_payload_b64": "…",             // npz-compressed; contains "height_mm" (uint16)
  "depth_format": "npz_uint16_height_mm"
  // else:
  // "depth_format": "none"
}
```

> `depth_payload_b64` is present **only** when the effective display is **HEIGHT** and a plane is available. It’s an **NPZ** with `height_mm` (uint16).

---

## Troubleshooting
- **`ModuleNotFoundError: cv2.aruco`** → Install `opencv-contrib-python` (not plain `opencv-python`).
- **No window / OpenCV HighGUI error** → Run in a desktop session (X11/Wayland/Windows), not a headless server; or disable GUI in code.
- **`ImportError: pyorbbecsdk`** → Install the Orbbec SDK Python bindings matching your Python version.
- **HEIGHT shows DEPTH instead** → Calibration is required or failed; ensure markers 1–4 are fully visible and well-lit.
- **Banner keeps appearing on ‘C’** → Camera doesn’t see all 4 IDs; verify **DICT_4X4_50** and the exact IDs 1,2,3,4.
- **No height outside ROI** → Height is computed but masked by the marker polygons and clamped to the configured range.

---

## Notes
- The app attempts to read camera **intrinsics** from the SDK; if available, they’re saved in `calibration_plane.json`.
- Marker regions are masked out of the height/depth visualization to reduce bias.
- Hole-filling (`MID_BLOB_*`) can be disabled by setting `MID_BLOB_ENABLE=0`.

---

## License
Internal tool. If you intend to distribute, add a proper license file.
