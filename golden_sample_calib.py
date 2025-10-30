#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbbec Viewer (RGB top | Depth/Height bottom)

Features
- T toggles between HEIGHT and DEPTH modes.
- P toggles probe (green crosshair) on/off; WASD/Arrow keys move it with press acceleration.
- Top-right: numeric readout (Height mm or Depth mm) only.
- Top-left: mode + context-aware controls/instructions + active mode range (min–max in mm).
- C triggers calibration capture (requires 4 ArUco markers with IDs 1–4 visible).
- If C is pressed without all 4 markers visible, a centered red banner appears for ~3s:
  "Need 4 visible markers. IDs 1,2,3,4"
- No mouse; no XYZ; no distance/point tools.

Controls
  T               Toggle mode (HEIGHT <-> DEPTH)
  P               Toggle probe on/off (green crosshair)
  W A S D         Move probe (accelerates while holding; max speed in ~2s)
  Arrow Keys      Move probe (accelerates while holding; max speed in ~2s)
  C               Start (re)calibration when 4 ArUco markers (IDs 1–4) are visible
  Q               Quit

Tunables (env)
  MID_PROBE_STEP                 Base step in pixels (default: 3)
  MID_ACCEL_GAIN                 Acceleration gain per second; if not set, derived from MID_ACCEL_TIME_TO_MAX_S
  MID_ACCEL_MAX_MULT             Max acceleration multiplier (default: 8.0)
  MID_ACCEL_TIME_TO_MAX_S        Seconds to reach max acceleration (default: 2.0)
  MID_ACCEL_REPEAT_GRACE_S       Max delay between repeats to continue hold (default: 0.12)
  MID_ACCEL_RELEASE_TIMEOUT_S    Stop hold after no repeat for this many seconds (default: 0.25)
  MID_DEPTH_VIZ_MIN_M            Depth visualization min (m)
  MID_DEPTH_VIZ_MAX_M            Depth visualization max (m)
  MID_HEIGHT_VIZ_MIN_M           Height visualization min (m)
  MID_HEIGHT_VIZ_MAX_M           Height visualization max (m)
"""

import os
import sys
import io
import cv2
import zmq
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime

# -----------------------------------------------------------------------------
# Optional extension path bootstrap
# -----------------------------------------------------------------------------
_deps = os.path.join(os.path.dirname(__file__), "_deps")
if os.path.isdir(_deps) and _deps not in sys.path:
    sys.path.insert(0, _deps)
ext_dir = os.path.join(_deps, "extensions")
if os.path.isdir(ext_dir):
    print(f"load extensions from {ext_dir}")

from pyorbbecsdk import Pipeline, Config, OBSensorType, AlignFilter, OBStreamType

# -----------------------------------------------------------------------------
# Configuration (env tunables)
# -----------------------------------------------------------------------------
VISUALIZATION = True
DEFAULT_WINDOW_SIZE = (1280, 720)

PUB_ENDPOINT = os.environ.get("MID_PUB", "tcp://127.0.0.1:5555")

USER_SET_CAL = int(os.environ.get("MID_CAL_FRAMES", "30"))
CAL_FRAMES = max(30, USER_SET_CAL)
LOG_LEVEL = os.environ.get("MID_LOG_LEVEL", "INFO")

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_IDS = {1, 2, 3, 4}

DEPTH_VIZ_MIN_M  = float(os.environ.get("MID_DEPTH_VIZ_MIN_M",  "0.20"))
DEPTH_VIZ_MAX_M  = float(os.environ.get("MID_DEPTH_VIZ_MAX_M",  "1.00"))
HEIGHT_VIZ_MIN_M = float(os.environ.get("MID_HEIGHT_VIZ_MIN_M", "0.001"))
HEIGHT_VIZ_MAX_M = float(os.environ.get("MID_HEIGHT_VIZ_MAX_M", "0.20"))

BLOB_ENABLE      = os.environ.get("MID_BLOB_ENABLE", "1") != "0"
BLOB_TOL_MM      = int(os.environ.get("MID_BLOB_TOL_MM", "10"))
BLOB_RING_PX     = int(os.environ.get("MID_BLOB_RING_PX", "3"))
BLOB_MIN_BORDER  = int(os.environ.get("MID_BLOB_MIN_BORDER", "24"))

CAL_MIN_VALID = int(os.environ.get("MID_CAL_MIN_VALID", "500"))

WINDOW_TITLE = "Orbbec Viewer (RGB ↑ | Depth/Height ↓)"
CAL_FILE = os.environ.get("MID_CAL_FILE", "calibration_plane.json")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("middle_service")

# -----------------------------------------------------------------------------
# ArUco detection (OpenCV 4.x compatibility)
# -----------------------------------------------------------------------------
try:
    _aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    _aruco_params = cv2.aruco.DetectorParameters()
    _aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

    def detect_aruco(gray):
        corners, ids, _ = _aruco_detector.detectMarkers(gray)
        return corners, ids
except Exception:
    logger.warning("Falling back to legacy cv2.aruco.detectMarkers API.")
    _aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
    _aruco_params = cv2.aruco.DetectorParameters_create()

    def detect_aruco(gray):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _aruco_dict, parameters=_aruco_params)
        return corners, ids

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def jpeg_b64(image_bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    import base64
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def npz_b64(**arrays):
    mem = io.BytesIO()
    np.savez_compressed(mem, **arrays)
    import base64
    return base64.b64encode(mem.getvalue()).decode("utf-8")

def draw_heatmap_mm(arr_mm, vmin_mm, vmax_mm):
    """Map uint16 depth/height (mm) into a JET colormap image (BGR)."""
    arr = arr_mm.astype(np.float32)
    mask_valid = (arr > 0)
    arr_clamped = np.clip(arr, vmin_mm, vmax_mm)
    denom = max(1.0, float(vmax_mm - vmin_mm))
    norm = ((arr_clamped - vmin_mm) / denom * 255.0).astype(np.uint8)
    norm[~mask_valid] = 0
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

def marker_mask_from_detections(shape_hw, corners, ids):
    """Return 8-bit mask (255 inside ArUco quads having IDs in ARUCO_IDS)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if ids is None:
        return mask
    ids_list = [int(x) for x in ids.ravel().tolist()]
    for i, mid in enumerate(ids_list):
        if mid in ARUCO_IDS:
            pts = np.array(corners[i], dtype=np.int32).reshape(-1, 2)
            cv2.fillConvexPoly(mask, pts, 255)
    return mask

def decode_color_frame(color_frame):
    """Decode color frame to BGR (handles raw BGR/YUYV/JPEG)."""
    c_w, c_h = color_frame.get_width(), color_frame.get_height()
    buf = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    if buf.size == c_w * c_h * 3:
        return buf.reshape((c_h, c_w, 3)).copy()
    if buf.size == c_w * c_h * 2:
        yuyv = buf.reshape((c_h, c_w, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode color frame (unknown format).")
    return bgr

def annotate_rgb(rgb, corners, ids, status_lines=None):
    """Overlay detected markers + status lines on RGB image (top-left only)."""
    out = rgb.copy()
    if ids is not None and len(ids) > 0:
        ids_list = ids.ravel().tolist()
        for i, mid in enumerate(ids_list):
            pts = np.array(corners[i], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
            cX, cY = pts.mean(axis=0).astype(int)
            cv2.putText(out, f"ID {int(mid)}", (cX - 20, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if status_lines:
        y = 30
        for line, color in status_lines:
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            y += 30
    return out

def resize_to_width(img, w):
    return cv2.resize(img, (w, int(img.shape[0] * (w / img.shape[1]))))

def make_disp_fit_window(win_name, top_bgr, bottom_bgr, use_window_rect=False):
    """Build stacked display image and report layout scales."""
    if use_window_rect:
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
            if win_w <= 0 or win_h <= 0:
                raise Exception("win rect not ready")
        except Exception:
            win_w = 1280
            win_h = resize_to_width(top_bgr, 1280).shape[0] + resize_to_width(bottom_bgr, 1280).shape[0]
    else:
        win_w = 1280
        win_h = resize_to_width(top_bgr, 1280).shape[0] + resize_to_width(bottom_bgr, 1280).shape[0]

    top_r = resize_to_width(top_bgr, win_w)
    bot_r = resize_to_width(bottom_bgr, win_w)
    tot_h = top_r.shape[0] + bot_r.shape[0]
    if tot_h > win_h:
        s = win_h / float(tot_h)
        new_w = max(1, int(win_w * s))
        top_r = resize_to_width(top_bgr, new_w)
        bot_r = resize_to_width(bottom_bgr, new_w)

    disp = np.vstack([top_r, bot_r])
    top_h = top_r.shape[0]
    scale_top = top_r.shape[1] / float(top_bgr.shape[1])
    scale_bot = bot_r.shape[1] / float(bottom_bgr.shape[1])
    return disp, top_h, scale_top, scale_bot

def convex_hull_from_markers(corners, ids):
    if ids is None or len(ids) == 0:
        return None
    pts = []
    ids_list = ids.ravel().tolist()
    for i, mid in enumerate(ids_list):
        if int(mid) in ARUCO_IDS:
            p = np.array(corners[i], dtype=np.float32).reshape(-1, 2)
            pts.append(p)
    if not pts:
        return None
    pts = np.vstack(pts)
    hull = cv2.convexHull(pts.astype(np.float32))
    return hull.reshape(-1, 2).astype(np.int32)

def polygon_mask(shape_hw, poly_xy):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly_xy is None or len(poly_xy) < 3:
        return mask
    cv2.fillConvexPoly(mask, poly_xy, 255)
    return mask

def try_get_intrinsics(profile):
    """Best-effort extraction of (fx, fy, cx, cy) from SDK profile."""
    cand_attrs = ["get_intrinsic", "get_intrinsics", "get_camera_intrinsics", "get_intrinsic_parameters"]
    intr = None
    for name in cand_attrs:
        try:
            fn = getattr(profile, name)
            intr = fn()
            break
        except Exception:
            continue
    if intr is None:
        return None
    for names in [
        ("fx", "fy", "cx", "cy"),
        ("Fx", "Fy", "Cx", "Cy"),
        ("f_x", "f_y", "c_x", "c_y"),
        ("ppx", "ppy", "fx", "fy"),
    ]:
        try:
            fx = float(getattr(intr, names[0]))
            fy = float(getattr(intr, names[1]))
            cx = float(getattr(intr, names[2]))
            cy = float(getattr(intr, names[3]))
            return fx, fy, cx, cy
        except Exception:
            continue
    try:
        fx = float(intr["fx"]); fy = float(intr["fy"]); cx = float(intr["cx"]); cy = float(intr["cy"])
        return fx, fy, cx, cy
    except Exception:
        return None

def fit_plane_svd(X, Y, Z, valid_mask):
    """Fit plane n·[X,Y,Z] + d = 0 using SVD on valid points; ensure n.z <= 0."""
    idx = np.where(valid_mask.ravel())[0]
    if idx.size < 50:
        return None
    A = np.stack([X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(Z).ravel()], axis=1)[idx, :]
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    plane = Vt[-1]
    n = plane[:3]
    d = plane[3]
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return None
    n = n / n_norm
    d = d / n_norm
    if n[2] > 0:
        n = -n
        d = -d
    return n, d

def load_calibration(path):
    """Load plane/intrinsics/ROI from JSON file if present."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        p = data.get("plane") or {}
        n = np.array([p["nx"], p["ny"], p["nz"]], dtype=np.float32)
        d = float(p["d"])
        intr = data.get("intrinsics") or {}
        fx = intr.get("fx"); fy = intr.get("fy"); cx = intr.get("cx"); cy = intr.get("cy")
        intrinsics = None
        if all(v is not None for v in (fx, fy, cx, cy)):
            intrinsics = (float(fx), float(fy), float(cx), float(cy))
        roi = data.get("roi_hull_px") or None
        if roi is not None:
            roi = np.array(roi, dtype=np.int32)
        return True, n, d, intrinsics, roi
    except Exception as e:
        logger.warning(f"Failed to load calibration from {path}: {e}")
        return False, None, None, None, None

def save_calibration(path, plane_n, plane_d, intrinsics, roi_hull_px):
    """Persist calibration JSON (plane, intrinsics, ROI, viz ranges)."""
    try:
        if intrinsics:
            fx, fy, cx, cy = intrinsics
        else:
            fx = fy = cx = cy = None
        if roi_hull_px is None:
            roi_list = []
        elif isinstance(roi_hull_px, np.ndarray):
            roi_list = roi_hull_px.reshape(-1, 2).astype(int).tolist()
        else:
            roi_list = [[int(x), int(y)] for (x, y) in roi_hull_px]
        data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "plane": {
                "nx": float(plane_n[0]),
                "ny": float(plane_n[1]),
                "nz": float(plane_n[2]),
                "d":  float(plane_d),
            },
            "intrinsics": {
                "fx": float(fx) if fx is not None else None,
                "fy": float(fy) if fy is not None else None,
                "cx": float(cx) if cx is not None else None,
                "cy": float(cy) if cy is not None else None,
            },
            "roi_hull_px": (roi_hull_px.reshape(-1, 2).astype(int).tolist()
                            if isinstance(roi_hull_px, np.ndarray) else (roi_hull_px or [])),
            "viz_ranges_m": {
                "depth":  {"min": float(DEPTH_VIZ_MIN_M),  "max": float(DEPTH_VIZ_MAX_M)},
                "height": {"min": float(HEIGHT_VIZ_MIN_M), "max": float(HEIGHT_VIZ_MAX_M)},
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration saved to {path}")
    except Exception as e:
        logger.exception("Failed to save calibration: %s", e)

def _touches_boundary(mask_bool):
    return mask_bool[0, :].any() or mask_bool[-1, :].any() or mask_bool[:, 0].any() or mask_bool[:, -1].any()

def fill_similar_holes_mm(img_mm, valid_mask, tol_mm=10, ring_px=3, min_border=24, skip_touching_boundary=True):
    """Fill small zero-value holes if border ring is consistent within tolerance."""
    img = img_mm.copy()
    holes = (img == 0) & valid_mask
    if not holes.any():
        return img
    holes_u8 = holes.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(holes_u8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    for lab in range(1, num_labels):
        comp = (labels == lab)
        if skip_touching_boundary and _touches_boundary(comp):
            continue
        comp_u8 = comp.astype(np.uint8)
        ring = cv2.dilate(comp_u8, kernel, iterations=1).astype(bool) & valid_mask
        border_vals = img[ring]
        border_vals = border_vals[border_vals > 0]
        if border_vals.size < min_border:
            continue
        lo = np.percentile(border_vals, 5)
        hi = np.percentile(border_vals, 95)
        if (hi - lo) <= tol_mm:
            fill_val = int(np.median(border_vals))
            img[comp] = fill_val
    return img

# -----------------------------------------------------------------------------
# Service
# -----------------------------------------------------------------------------
class MiddleService:
    """Main runtime: camera pipeline, visualization, probe controls, ZMQ PUB."""

    def __init__(self):
        # Calibration state
        self.has_plane = False
        self.plane_n = None
        self.plane_d = None
        self.intrinsics = None
        self.roi_hull_px = None

        # Calibration capture control
        self._accum_lock = threading.Lock()
        self._accum_frames = []
        self._calibration_mode = False
        self._arm_calibration = False
        self._await_markers = False

        # Meshgrid for per-pixel coordinates
        self._uv_ready = False
        self._u = None
        self._v = None

        # UI / display state
        self.gui = bool(VISUALIZATION)
        self._last_key = -1
        self._disp_frame = None
        self._disp_lock = threading.Lock()

        # Probe (toggle) + movement acceleration
        self.probe_active = False
        self.cursor_uv = None                                 # (u, v) in depth alignment space
        self.probe_step = int(os.environ.get("MID_PROBE_STEP", "3"))
        self.last_probe_sample = None

        # --- Acceleration tuning: reach max speed in ~2s by default ---
        self.accel_max_mult = float(os.environ.get("MID_ACCEL_MAX_MULT", "8.0"))
        accel_time_to_max_s = float(os.environ.get("MID_ACCEL_TIME_TO_MAX_S", "2.0"))
        if "MID_ACCEL_GAIN" in os.environ:
            self.accel_gain = float(os.environ.get("MID_ACCEL_GAIN", "3.5"))
        else:
            # Derive gain so that: 1 + gain * t_max = accel_max_mult  ->  gain = (M-1)/t_max
            self.accel_gain = (self.accel_max_mult - 1.0) / max(0.001, accel_time_to_max_s)

        self.accel_repeat_grace_s = float(os.environ.get("MID_ACCEL_REPEAT_GRACE_S", "0.12"))
        self.accel_release_timeout_s = float(os.environ.get("MID_ACCEL_RELEASE_TIMEOUT_S", "0.25"))

        self.move_is_holding = False
        self.move_dir_unit = (0, 0)      # (-1/0/1, -1/0/1)
        self.move_pre_hold_start_t = 0.0
        self.move_hold_start_t = 0.0
        self.move_last_event_t = 0.0

        # Latest arrays for sampling & layout
        self._disp_meta = None
        self._last_depth_mm = None
        self._last_height_mm = None
        self._last_mode = "depth"        # effective mode actually displayed

        # User-selected mode (T toggles). Default: HEIGHT if calibration exists, else DEPTH.
        self.viz_mode = "depth"

        # Flash banner (transient message)
        self._flash_msg = None
        self._flash_color = (0, 0, 255)          # red
        self._flash_start_t = 0.0
        self._flash_duration = 0.0

        # Messaging
        self.zctx = zmq.Context.instance()
        self.pub = self.zctx.socket(zmq.PUB)
        self.pub.bind(PUB_ENDPOINT)
        logger.info(f"PUB bound at {PUB_ENDPOINT}")

        # Camera pipeline
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter = None
        self.color_profile = None
        self.depth_profile = None
        self._setup_camera()

        # Load prior calibration if available
        ok, n, d, intr, roi = load_calibration(CAL_FILE)
        if ok:
            self.plane_n, self.plane_d = n, d
            self.roi_hull_px = roi
            self.has_plane = True
            if intr:
                self.intrinsics = intr
            self.viz_mode = "height"
            logger.info("Calibration loaded; starting in HEIGHT mode.")
        else:
            self.viz_mode = "depth"
            logger.info("No calibration file; starting in DEPTH mode.")

        # Threading
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # ---------------------- Lifecycle ----------------------
    def _post_key(self, k: int):
        self._last_key = k

    def _consume_key(self) -> int:
        k = self._last_key
        self._last_key = -1
        return k

    def _setup_camera(self):
        """Configure and start Orbbec pipeline; align depth->color."""
        try:
            plist = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            self.color_profile = plist.get_default_video_stream_profile()
            self.config.enable_stream(self.color_profile)

            plist = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            self.depth_profile = plist.get_default_video_stream_profile()
            self.config.enable_stream(self.depth_profile)

            self.pipeline.enable_frame_sync()
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self.pipeline.start(self.config)
            logger.info("Pipeline started (aligned depth -> color).")

            if self.intrinsics is None:
                self.intrinsics = try_get_intrinsics(self.depth_profile)
                if self.intrinsics:
                    fx, fy, cx, cy = self.intrinsics
                    logger.info(f"Depth intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                else:
                    logger.warning("Depth intrinsics not available yet; will try again.")
        except Exception as e:
            logger.exception("Failed to configure/start Orbbec pipeline: %s", e)
            raise

    def reset_calibration(self):
        """Clear plane/ROI and stop accumulation; keep intrinsics."""
        self.has_plane = False
        self.plane_n = None
        self.plane_d = None
        self.roi_hull_px = None
        with self._accum_lock:
            self._accum_frames = []
        self._calibration_mode = False
        self._arm_calibration = False
        self._await_markers = False
        logger.info("Calibration reset.")

    def start(self):
        self.thread.start()
        logger.info("Main loop started.")

    def stop(self):
        self.stop_evt.set()
        self.thread.join(timeout=3.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.pub.close(0)
        self.zctx.term()
        logger.info("Stopped.")

    # ---------------------- Flash banner ----------------------
    def _flash(self, msg: str, color=(0, 0, 255), duration_s: float = 3.0):
        """Trigger a transient banner message."""
        self._flash_msg = msg
        self._flash_color = color
        self._flash_start_t = time.monotonic()
        self._flash_duration = max(0.1, float(duration_s))

    def _draw_flash_if_any(self, img):
        """Overlay fading, centered banner if active."""
        if not self._flash_msg:
            return
        elapsed = time.monotonic() - self._flash_start_t
        if elapsed >= self._flash_duration:
            # Clear flash
            self._flash_msg = None
            return
        # Alpha decreases linearly over the duration
        alpha = max(0.0, 1.0 - (elapsed / self._flash_duration))

        # Compute text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.9
        th = 2
        (tw, th_text), _ = cv2.getTextSize(self._flash_msg, font, fs, th)

        # Centered rectangle behind the text
        pad = 16
        box_w = tw + pad * 2
        box_h = th_text + pad * 2

        H, W = img.shape[:2]
        x0 = max(0, (W - box_w) // 2)
        y0 = max(0, (H - box_h) // 2)
        x1 = min(W - 1, x0 + box_w)
        y1 = min(H - 1, y0 + box_h)

        # Build overlay
        overlay = img.copy()
        # Semi-transparent dark box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        # Optional border (subtle)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 40, 40), 2)

        # Text coordinates (baseline inside the box)
        text_x = x0 + pad
        text_y = y0 + pad + th_text

        # Text with shadow + colored foreground
        cv2.putText(overlay, self._flash_msg, (text_x, text_y), font, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(overlay, self._flash_msg, (text_x, text_y), font, fs, self._flash_color, th + 1, cv2.LINE_AA)

        # Blend using alpha (cap overlay weight to avoid over-darkening)
        img[:] = cv2.addWeighted(img, 1.0, overlay, min(0.85, alpha), 0)

    # ---------------------- Geometry helpers ----------------------
    def _ensure_uv_grid(self, w, h):
        """Prepare u,v meshgrid for current depth resolution."""
        if self._uv_ready and self._u.shape == (h, w):
            return
        u = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        v = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        self._u, self._v = u, v
        self._uv_ready = True
        logger.info(f"UV grid prepared for {w}x{h}")

    def _source_to_disp_top(self, u, v):
        meta = self._disp_meta
        if meta is None:
            return None
        s_top = meta["scale_top"]
        return int(u * s_top), int(v * s_top)

    def _source_to_disp_bot(self, u, v):
        meta = self._disp_meta
        if meta is None:
            return None
        top_h = meta["top_h"]
        s_bot = meta["scale_bot"]
        return int(u * s_bot), int(v * s_bot) + int(top_h)

    # ---------------------- Sampling & movement ----------------------
    def _sample_at_uv(self, u, v):
        """Return current-mode value (mm) at (u,v): height in height mode else depth."""
        depth_raw = int(self._last_depth_mm[v, u]) if self._last_depth_mm is not None else 0
        height_val = int(self._last_height_mm[v, u]) if self._last_height_mm is not None else None
        if self._last_mode == "height" and height_val is not None:
            mode_val_mm = height_val
            label = "Height"
        else:
            mode_val_mm = depth_raw
            label = "Depth"
        return {"value_mm": mode_val_mm, "label": label}

    @staticmethod
    def _unit_dir(du, dv):
        su = 0 if du == 0 else (1 if du > 0 else -1)
        sv = 0 if dv == 0 else (1 if dv > 0 else -1)
        return su, sv

    def _dynamic_step(self, now):
        hold_s = max(0.0, now - self.move_hold_start_t)
        mult = min(self.accel_max_mult, 1.0 + self.accel_gain * hold_s)
        return max(1, int(round(self.probe_step * mult)))

    def _apply_move(self, step, w, h):
        if not self.probe_active:
            return
        if self.cursor_uv is None:
            self.cursor_uv = (w // 2, h // 2)
        du = self.move_dir_unit[0] * step
        dv = self.move_dir_unit[1] * step
        u = int(np.clip(self.cursor_uv[0] + du, 0, w - 1))
        v = int(np.clip(self.cursor_uv[1] + dv, 0, h - 1))
        self.cursor_uv = (u, v)

    # ---------------------- UI loop (keyboard-only) ----------------------
    def ui_loop(self):
        if not self.gui:
            return
        try:
            flags = cv2.WINDOW_NORMAL
            if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
                flags |= cv2.WINDOW_GUI_EXPANDED
            cv2.namedWindow(WINDOW_TITLE, flags)
            cv2.resizeWindow(WINDOW_TITLE, DEFAULT_WINDOW_SIZE[0], DEFAULT_WINDOW_SIZE[1])
        except cv2.error as e:
            logger.error(f"Visualization init failed (window create): {e}")
            self.stop_evt.set()
            return

        while not self.stop_evt.is_set() and self.thread.is_alive():
            frame = None
            with self._disp_lock:
                frame = self._disp_frame
            if frame is not None:
                cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._post_key(key)
                if key in (ord('q'), ord('Q')):
                    self.stop_evt.set()
                    break
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # ---------------------- Main loop ----------------------
    def _loop(self):
        first_log = True
        while not self.stop_evt.is_set():
            try:
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue
                frames = self.align_filter.process(frames)
                if not frames:
                    continue
                fs = frames.as_frame_set()
                color_frame = fs.get_color_frame()
                depth_frame = fs.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                d_w, d_h = depth_frame.get_width(), depth_frame.get_height()
                color = decode_color_frame(color_frame)

                if first_log:
                    logger.info(f"RGB {color.shape[1]}x{color.shape[0]} | Depth(aligned) {d_w}x{d_h}")
                    logger.info("Controls: T mode | P probe | WASD/Arrows move (accelerates) | C calibrate | Q quit")
                    first_log = False

                # Intrinsics (best-effort)
                if self.intrinsics is None:
                    self.intrinsics = try_get_intrinsics(self.depth_profile)
                    if self.intrinsics:
                        fx, fy, cx, cy = self.intrinsics
                        logger.info(f"Depth intrinsics acquired: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

                self._ensure_uv_grid(d_w, d_h)

                # Depth in mm
                depth_scale_mm_per_unit = float(depth_frame.get_depth_scale())
                depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((d_h, d_w))
                depth_mm = (depth_raw.astype(np.float32) * depth_scale_mm_per_unit).astype(np.uint16)

                # ArUco detection / masks
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                corners, ids = detect_aruco(gray)
                ids_list = ids.ravel().tolist() if ids is not None else []
                ids_set = set(int(x) for x in ids_list)
                seen_all_four = set(ARUCO_IDS).issubset(ids_set)

                marker_mask = marker_mask_from_detections(depth_mm.shape, corners, ids)
                hull = convex_hull_from_markers(corners, ids)

                # ---------------- Keyboard handling ----------------
                key = self._consume_key()
                now = time.monotonic()

                if key in (ord('q'), ord('Q')):
                    break

                # Toggle mode
                if key in (ord('t'), ord('T')):
                    self.viz_mode = "height" if self.viz_mode == "depth" else "depth"

                # Calibration request
                if key in (ord('c'), ord('C')):
                    # If all markers visible, arm calibration; else show centered banner.
                    if seen_all_four:
                        self._arm_calibration = True
                        self._calibration_mode = True
                        with self._accum_lock:
                            self._accum_frames = []
                        self._await_markers = False
                        logger.info("Calibration started (capturing %d frames)...", CAL_FRAMES)
                    else:
                        self._await_markers = True
                        self._flash(
                            "Need 4 visible markers. IDs 1,2,3,4",
                            color=(0, 0, 255),  # red text
                            duration_s=3.0
                        )

                # Toggle probe
                if key in (ord('p'), ord('P')):
                    if not self.probe_active or self.cursor_uv is None:
                        self.probe_active = True
                        self.cursor_uv = (d_w // 2, d_h // 2)
                        self.move_is_holding = False
                    else:
                        self.probe_active = False
                        self.cursor_uv = None
                        self.last_probe_sample = None
                        self.move_is_holding = False

                # Arrow keycodes (Qt backend): left=81, up=82, right=83, down=84
                ARROW_LEFT, ARROW_UP, ARROW_RIGHT, ARROW_DOWN = 81, 82, 83, 84
                move_map = {
                    ord('a'): (-self.probe_step, 0),
                    ord('d'): ( self.probe_step, 0),
                    ord('w'): (0, -self.probe_step),
                    ord('s'): (0,  self.probe_step),
                    ARROW_LEFT: (-self.probe_step, 0),
                    ARROW_RIGHT:( self.probe_step, 0),
                    ARROW_UP:   (0, -self.probe_step),
                    ARROW_DOWN: (0,  self.probe_step),
                }

                # Discrete movement key pressed?
                if key in move_map and self.probe_active:
                    du, dv = move_map[key]
                    dir_unit = self._unit_dir(du, dv)
                    same_dir = (dir_unit == self.move_dir_unit)
                    within_grace = (now - self.move_last_event_t) <= self.accel_repeat_grace_s

                    if same_dir and within_grace:
                        # continue hold
                        if not self.move_is_holding:
                            self.move_is_holding = True
                            self.move_hold_start_t = self.move_pre_hold_start_t
                        self.move_last_event_t = now
                    else:
                        # fresh press / new direction
                        self.move_is_holding = False
                        self.move_dir_unit = dir_unit
                        self.move_pre_hold_start_t = now
                        self.move_last_event_t = now
                        self._apply_move(self.probe_step, d_w, d_h)  # responsive tap

                # Continuous movement while holding (accelerated)
                if self.probe_active and self.move_is_holding:
                    if (now - self.move_last_event_t) <= self.accel_release_timeout_s:
                        step = self._dynamic_step(now)
                        self._apply_move(step, d_w, d_h)
                    else:
                        self.move_is_holding = False
                # ----------------------------------------------------

                # Precompute viz ranges in mm for display text
                dmin_mm = int(DEPTH_VIZ_MIN_M * 1000)
                dmax_mm = int(DEPTH_VIZ_MAX_M * 1000)
                hmin_mm = int(HEIGHT_VIZ_MIN_M * 1000)
                hmax_mm = int(HEIGHT_VIZ_MAX_M * 1000)

                # --- Compute height (if possible) ---
                z_m = depth_mm.astype(np.float32) / 1000.0
                height_mm = None
                if self.intrinsics is not None and self.plane_n is not None:
                    fx, fy, cx, cy = self.intrinsics
                    X = (self._u - cx) / fx * z_m
                    Y = (self._v - cy) / fy * z_m
                    Z = z_m
                    height_m = (self.plane_n[0] * X + self.plane_n[1] * Y + self.plane_n[2] * Z + self.plane_d)
                    height_m[z_m <= 0] = 0.0
                    if marker_mask.any():
                        height_m = height_m.copy()
                        height_m[marker_mask == 255] = 0.0
                    height_mm_raw = np.clip(np.round(height_m * 1000.0), 0, 65535).astype(np.uint16)
                    if BLOB_ENABLE:
                        valid_mask_height = (height_mm_raw > 0) & (marker_mask == 0)
                        height_mm = fill_similar_holes_mm(
                            height_mm_raw, valid_mask_height,
                            tol_mm=BLOB_TOL_MM, ring_px=BLOB_RING_PX, min_border=BLOB_MIN_BORDER
                        )
                    else:
                        height_mm = height_mm_raw
                    # Clamp to viz range
                    in_range = (height_mm >= hmin_mm) & (height_mm <= hmax_mm)
                    height_mm = np.where(in_range, height_mm, 0).astype(np.uint16)

                # --- Calibration accumulation / solve plane ---
                if self._calibration_mode and self._arm_calibration and seen_all_four:
                    with self._accum_lock:
                        if len(self._accum_frames) < CAL_FRAMES:
                            self._accum_frames.append(depth_mm.copy())
                        if len(self._accum_frames) >= CAL_FRAMES:
                            med_mm = np.median(np.stack(self._accum_frames, axis=0), axis=0).astype(np.uint16)
                            final_hull = hull if (hull is not None and len(hull) >= 3) else None
                            roi_mask = polygon_mask(depth_mm.shape, final_hull) > 0 if final_hull is not None else np.ones_like(med_mm, dtype=bool)
                            valid = (med_mm > 0) & roi_mask & (marker_mask == 0)
                            num_valid = int(np.count_nonzero(valid))
                            if num_valid < CAL_MIN_VALID:
                                logger.warning("Calibration failed: only %d valid ROI points (need >= %d)", num_valid, CAL_MIN_VALID)
                                self._calibration_mode = False
                                self._arm_calibration = False
                                self._accum_frames = []
                                self._flash("Calibration failed: insufficient valid points in ROI. Adjust view and retry.", (0, 0, 255), 3.0)
                            else:
                                # Fit plane
                                if self.intrinsics is not None:
                                    fx, fy, cx, cy = self.intrinsics
                                    med_m = med_mm.astype(np.float32) / 1000.0
                                    X = (self._u - cx) / fx * med_m
                                    Y = (self._v - cy) / fy * med_m
                                    Z = med_m
                                    plane = fit_plane_svd(X, Y, Z, valid)
                                    if plane is None:
                                        self._calibration_mode = False
                                        self._arm_calibration = False
                                        self._accum_frames = []
                                        self._flash("Calibration failed: plane fit could not converge. Reposition and retry.", (0, 0, 255), 3.0)
                                    else:
                                        self.plane_n, self.plane_d = plane
                                        self.roi_hull_px = final_hull
                                        save_calibration(CAL_FILE, self.plane_n, self.plane_d, self.intrinsics, self.roi_hull_px)
                                        self.has_plane = True
                                        self._calibration_mode = False
                                        self._arm_calibration = False
                                        self._accum_frames = []
                                        self._flash("Calibration complete. Height mode enabled.", (0, 200, 0), 2.0)

                # --- Choose effective display mode (fallback if height unavailable) ---
                selected_mode = self.viz_mode  # user's choice
                if selected_mode == "height" and height_mm is None:
                    effective_mode = "depth"
                else:
                    effective_mode = selected_mode

                # --- Build heatmap for effective mode ---
                if effective_mode == "depth":
                    if BLOB_ENABLE:
                        valid_mask_depth = (depth_mm > 0) & (marker_mask == 0)
                        depth_mm_filled = fill_similar_holes_mm(
                            depth_mm, valid_mask_depth, tol_mm=BLOB_TOL_MM,
                            ring_px=BLOB_RING_PX, min_border=BLOB_MIN_BORDER
                        )
                    else:
                        depth_mm_filled = depth_mm
                    in_range = (depth_mm_filled >= dmin_mm) & (depth_mm_filled <= dmax_mm)
                    depth_mm_viz = np.where(in_range, depth_mm_filled, 0).astype(np.uint16)
                    if marker_mask.any():
                        depth_mm_viz[marker_mask == 255] = 0
                    heat = draw_heatmap_mm(depth_mm_viz, dmin_mm, dmax_mm)
                else:
                    heat = draw_heatmap_mm(height_mm, hmin_mm, hmax_mm)

                # Compose top-left instructions (mode-aware) + range for active (effective) mode
                instruct = []
                if selected_mode == "height":
                    if height_mm is None:
                        instruct.append(("MODE: HEIGHT (calibration required - currently showing DEPTH)", (0, 165, 255)))
                    else:
                        instruct.append(("MODE: HEIGHT", (180, 255, 180)))
                else:
                    instruct.append(("MODE: DEPTH", (180, 255, 180)))

                # Add active range line (based on effective display mode)
                if effective_mode == "depth":
                    instruct.append((f"Range: {dmin_mm}–{dmax_mm} mm", (200, 255, 200)))
                else:
                    instruct.append((f"Range: {hmin_mm}–{hmax_mm} mm", (200, 255, 200)))

                # Calibration capture progress hint
                if self._calibration_mode and self._arm_calibration:
                    with self._accum_lock:
                        left = max(0, CAL_FRAMES - len(self._accum_frames))
                    instruct.append((f"CALIBRATING: capturing frames... remaining: {left}", (255, 255, 255)))

                instruct.append(("Controls: T mode | P probe | WASD/Arrows move (accelerates) | C calibrate | Q quit",
                                 (200, 200, 255)))

                # Annotate RGB with markers + instructions (top-left only)
                overlay = annotate_rgb(color, corners, ids, status_lines=instruct)

                # Pack to single display image
                disp, top_h, scale_top, scale_bot = make_disp_fit_window(WINDOW_TITLE, overlay, heat, use_window_rect=False)

                # Update shared state for sampling & drawing
                with self._disp_lock:
                    self._disp_meta = {"top_h": int(top_h), "scale_top": float(scale_top), "scale_bot": float(scale_bot)}
                    self._last_depth_mm = depth_mm.copy()
                    self._last_height_mm = (height_mm.copy() if height_mm is not None else None)
                    self._last_mode = effective_mode

                # Live probe sampling
                if self.probe_active and self.cursor_uv is not None:
                    u = int(np.clip(self.cursor_uv[0], 0, d_w - 1))
                    v = int(np.clip(self.cursor_uv[1], 0, d_h - 1))
                    self.cursor_uv = (u, v)
                    self.last_probe_sample = self._sample_at_uv(u, v)
                else:
                    self.last_probe_sample = None

                # --------- Draw overlays: probe (green) + RIGHT-TOP readout + centered flash ----------
                disp_annot = disp.copy()

                def draw_text_with_shadow(img, text, org, font_scale=0.8, thickness=2):
                    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
                    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                def draw_top_right(img, text, y_pix):
                    margin = 10
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    x = img.shape[1] - margin - tw
                    y = max(th + margin, y_pix)
                    draw_text_with_shadow(img, text, (x, y))

                if self.probe_active and self.cursor_uv is not None:
                    cu, cvv = int(self.cursor_uv[0]), int(self.cursor_uv[1])
                    # Crosshair on both panes
                    for pt in (self._source_to_disp_top(cu, cvv), self._source_to_disp_bot(cu, cvv)):
                        if pt is None:
                            continue
                        cv2.drawMarker(disp_annot, pt, (50, 255, 50), markerType=cv2.MARKER_CROSS,
                                       markerSize=18, thickness=2)
                        cv2.circle(disp_annot, pt, 3, (50, 255, 50), -1)

                    if self.last_probe_sample is not None:
                        text = f"{self.last_probe_sample['label']}: {self.last_probe_sample['value_mm']} mm"
                        # Right-top on top pane
                        draw_top_right(disp_annot, text, y_pix=30)
                        # Right-top on bottom pane (offset by top_h)
                        draw_top_right(disp_annot, text, y_pix=int(top_h) + 30)

                # Centered flash banner (if any)
                self._draw_flash_if_any(disp_annot)
                # ---------------------------------------------------------------------------

                # Publish composed frame to UI
                if self.gui:
                    with self._disp_lock:
                        self._disp_frame = disp_annot

                # PUB payload (frame+viz info)
                def _roi_list(x):
                    if x is None:
                        return None
                    return x if isinstance(x, list) else np.asarray(x).astype(int).tolist()

                payload = {
                    "type": "frame",
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "frame_name": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                    "rgb_jpeg_b64": jpeg_b64(color),
                    "aligned_to_color": True,
                    "aruco": {
                        "ids": ids_list,
                        "masked_pixels": int(np.count_nonzero(marker_mask))
                    },
                    "meta": {
                        "has_plane": bool(self.plane_n is not None),
                        "selected_mode": self.viz_mode,
                        "effective_mode": self._last_mode,
                        "viz_ranges_m": {
                            "depth":  {"min": float(DEPTH_VIZ_MIN_M),  "max": float(DEPTH_VIZ_MAX_M)},
                            "height": {"min": float(HEIGHT_VIZ_MIN_M), "max": float(HEIGHT_VIZ_MAX_M)},
                        },
                        "depth_scale_mm_per_unit": float(depth_scale_mm_per_unit),
                        "blob_fill": {
                            "enabled": bool(BLOB_ENABLE),
                            "tol_mm": int(BLOB_TOL_MM),
                            "ring_px": int(BLOB_RING_PX),
                            "min_border": int(BLOB_MIN_BORDER),
                        },
                        "acceleration": {
                            "max_mult": float(self.accel_max_mult),
                            "gain_per_s": float(self.accel_gain)
                        }
                    }
                }

                payload["rgb_annotated_jpeg_b64"] = jpeg_b64(overlay)
                payload["visualization"] = {"enabled": True}
                if self._last_mode == "height" and self.plane_n is not None and height_mm is not None:
                    payload["depth_payload_b64"] = npz_b64(height_mm=height_mm)
                    payload["depth_format"] = "npz_uint16_height_mm"
                    payload["calibration"] = {
                        "plane": {"nx": float(self.plane_n[0]), "ny": float(self.plane_n[1]),
                                  "nz": float(self.plane_n[2]), "d": float(self.plane_d)},
                        "roi_hull_px": _roi_list(self.roi_hull_px),
                    }
                    if self.intrinsics is not None:
                        fx, fy, cx, cy = self.intrinsics
                        payload["calibration"]["intrinsics"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
                    payload["visualization"]["heatmap_jpeg_b64"] = jpeg_b64(heat)
                else:
                    payload["depth_format"] = "none"
                    payload["visualization"]["heatmap_jpeg_b64"] = jpeg_b64(heat)

                self.pub.send_json(payload, flags=0)

            except Exception as e:
                logger.exception("Processing loop error: %s", e)
                time.sleep(0.01)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    svc = MiddleService()
    try:
        svc.start()
        svc.ui_loop()
        svc.thread.join(timeout=3.0)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()

