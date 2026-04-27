"""
Hybrid DVS+RGB Pupil Detector: RITnet (~120Hz) + TDTracker (~1000Hz)

Thread architecture
-------------------
  main thread   : RITnet per RGB frame → IPC publish + calibration pairing
  dvs_thread    : 1ms EventStreamSlicer → BinarRep → infer_queue → publish_state
  infer_thread  : TDTracker on best available CUDA/CPU device, seq=8 → latest_result
  publish_thread: 1ms timer (sleep + short busy-wait) → IPC "pupil.{id}.dvs"
  viz_thread    : cv2 window — live DVS events + predicted pupil dot

Coordinate conventions
----------------------
  DVS native   : integer pixel coords, y=0 at TOP of sensor
  DVS-norm     : (x, y) ∈ [0,1], y=0 at TOP  (divide by sensor dims)
  Pupil norm   : (x, y) ∈ [0,1], y=0 at BOTTOM  → flip: y_pupil = 1 − y_dvs

  DVSRGBCalibrator receives (dvs_norm, rgb_norm) where dvs_norm is un-flipped.
  The resulting homography H maps un-flipped DVS-norm → Pupil convention.
  After calibrator.apply(), the output is already in Pupil convention — no extra flip.
"""

import logging
import os
import queue
import threading
import time
import types
from collections import deque
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import torch
from gl_utils import draw_circle_filled_func_builder
from pyglui.cygl.utils import RGBA

from .detector_2d_plugin import Detector2DPlugin
from .dvs_models.TDTracker import Model
from .dvs_metrics import decode_batch_sa_simdr
from .dvs_rgb_calibrator import DVSRGBCalibrator

logger = logging.getLogger(__name__)

_draw_circle_filled = draw_circle_filled_func_builder()


# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

DAVIS_WIDTH  = 346
DAVIS_HEIGHT = 260

# TDTracker input resolution (must match SA-SimDR output bins: x→80, y→60)
DVS_W = 80
DVS_H = 60


# ---------------------------------------------------------------------------
# Algorithm parameters
# ---------------------------------------------------------------------------

# Slicer: one callback per TIME_WINDOW_US microseconds → 1000 callbacks/s
TIME_WINDOW_US  = 1_000
# Sliding window depth fed to TDTracker each inference call.
# Bidirectional GRU/Mamba requires the full sequence for correct backward context.
SEQUENCE_LENGTH = 8
# Number of time bins in one BinarRep frame (values 0 = 2^0 + ... = 15)
N_TIME_BINS = 4

# BinarRep backend:
#   "numpy" = low-overhead local implementation
#   "tonic" = reference-style tonic.transforms pipeline
#
# To select from main.py before eye processes start:
#   os.environ["PUPIL_HYBRID_BINAREP_BACKEND"] = "tonic"
#
# To select directly in this plugin, change BINAREP_BACKEND_DEFAULT below.
BINAREP_BACKEND_ENV = "PUPIL_HYBRID_BINAREP_BACKEND"
BINAREP_BACKEND_DEFAULT = "numpy"

# Calibration pairing: maximum allowed gap between an RGB and DVS timestamp
MAX_PAIR_GAP_SEC = 0.015   # 15ms

# DVS position EMA filter (applied before IPC publish and calibration pairing)
CONF_THRESHOLD  = 0.60   # EMA update only when confidence ≥ this value
EMA_ALPHA       = 0.15   # smoothing factor (higher = more responsive, noisier)
MAX_JUMP        = 0.08   # outlier gate in normalized coords
JUMP_RESET_CNT  = 10     # reset EMA after this many consecutive blocked frames (saccade recovery)

# Staleness: skip DVS slices if the camera backlog exceeds this wall-clock lag
MAX_LAG_SEC = 0.05

# Clock sync: re-anchor DVS↔Pupil timestamps this often to correct drift
SYNC_UPDATE_INTERVAL = 30.0   # seconds

# RITnet anchoring: slowly correct mapped DVS coordinates toward the absolute
# RITnet norm_pos when both observations are temporally aligned.
RITNET_ANCHOR_CONF_THRESHOLD = 0.50
ANCHOR_OFFSET_ALPHA          = 0.05
ANCHOR_MAX_OFFSET            = 0.25
DVS_STATE_HOLD_SEC           = 0.05


# ---------------------------------------------------------------------------
# BinarRep frame encoding
# ---------------------------------------------------------------------------

_BINAREP_MAX = float((1 << N_TIME_BINS) - 1)   # 15.0 — used for viz normalisation
_TONIC_BINAREP_PIPELINES = {}


def _resolve_binarep_backend(value: Optional[str] = None) -> str:
    backend = value or os.environ.get(BINAREP_BACKEND_ENV, BINAREP_BACKEND_DEFAULT)
    backend = backend.strip().lower()
    if backend not in {"numpy", "tonic"}:
        logger.warning(
            f"[Hybrid] Unknown BinarRep backend '{backend}', falling back to numpy."
        )
        return "numpy"
    return backend


def _to_binarep_numpy(
    xs:   np.ndarray,
    ys:   np.ndarray,
    pols: np.ndarray,
    ts:   np.ndarray,
    W:    int = DVS_W,
    H:    int = DVS_H,
    n_bits: int = N_TIME_BINS,
) -> np.ndarray:
    """
    Convert raw events in one time window to a BinarRep frame.

    Self-contained implementation of the frame encoding used during training.

    Returns
    -------
    frame : np.ndarray, shape (2, H, W), dtype float32
        channel 0 = OFF events (polarity 0)
        channel 1 = ON  events (polarity 1)
        values:  0–15  (sum of 2**b for each occupied bin, NOT divided by 15)

    WARNING: Do NOT divide by 15 before passing to TDTracker.
    The model was trained on unnormalised values.
    """
    frame = np.zeros((2, H, W), dtype=np.float32)
    if len(ts) == 0:
        return frame

    t0       = int(ts[0])
    duration = max(int(ts[-1]) - t0, 1)
    bins     = np.floor((ts - t0) / (duration + 1) * n_bits).astype(np.int32)
    np.clip(bins, 0, n_bits - 1, out=bins)

    for b in range(n_bits):
        mask_b = bins == b
        if not mask_b.any():
            continue
        bit = float(1 << b)   # 1, 2, 4, 8 — matches ToBinaRep
        for pol in (0, 1):
            m = mask_b & (pols == pol)
            if m.any():
                hit = np.zeros((H, W), dtype=np.bool_)
                hit[ys[m], xs[m]] = True
                frame[pol] += hit * bit

    return frame


def _to_binarep_tonic(
    xs:   np.ndarray,
    ys:   np.ndarray,
    pols: np.ndarray,
    ts:   np.ndarray,
    W:    int = DVS_W,
    H:    int = DVS_H,
    n_bits: int = N_TIME_BINS,
) -> np.ndarray:
    """Reference-style BinarRep path using tonic, selected only when requested."""
    import tonic.transforms as transforms

    key = (W, H, n_bits)
    pipeline = _TONIC_BINAREP_PIPELINES.get(key)
    if pipeline is None:
        pipeline = transforms.Compose([
            transforms.ToFrame(sensor_size=(W, H, 2), n_time_bins=n_bits),
            transforms.ToBinaRep(n_frames=1, n_bits=n_bits),
        ])
        _TONIC_BINAREP_PIPELINES[key] = pipeline

    length = len(ts)
    dtype = [("x", "<i8"), ("y", "<i8"), ("t", "<i8"), ("p", "<i8")]
    ev = np.zeros(length, dtype=dtype)
    ev["x"] = xs.astype(np.int64)
    ev["y"] = ys.astype(np.int64)
    ev["t"] = ts.astype(np.int64)
    ev["p"] = pols.astype(np.int64)

    frame = pipeline(ev)
    if frame.ndim == 4:
        frame = frame[0]
    return np.asarray(frame, dtype=np.float32)


def _to_binarep(
    xs:   np.ndarray,
    ys:   np.ndarray,
    pols: np.ndarray,
    ts:   np.ndarray,
    backend: str,
    W:    int = DVS_W,
    H:    int = DVS_H,
    n_bits: int = N_TIME_BINS,
) -> np.ndarray:
    if backend == "tonic":
        return _to_binarep_tonic(xs, ys, pols, ts, W=W, H=H, n_bits=n_bits)
    return _to_binarep_numpy(xs, ys, pols, ts, W=W, H=H, n_bits=n_bits)


# ---------------------------------------------------------------------------
# Lightweight rate counter
# ---------------------------------------------------------------------------

class _RateCounter:
    """Accumulate ticks; return Hz once per second, otherwise None."""

    __slots__ = ("_n", "_t0")

    def __init__(self):
        self._n  = 0
        self._t0 = time.monotonic()

    def tick(self) -> Optional[float]:
        self._n += 1
        now     = time.monotonic()
        elapsed = now - self._t0
        if elapsed >= 1.0:
            hz       = self._n / elapsed
            self._n  = 0
            self._t0 = now
            return hz
        return None


# ---------------------------------------------------------------------------
# DVS window visualiser
# ---------------------------------------------------------------------------

_VIZ_SCALE = 5   # 80×60 → 400×300


def _render_dvs(frame: np.ndarray, result: Optional[tuple]) -> np.ndarray:
    """
    BGR image from one BinarRep frame.

    Green channel = OFF events (channel 0 / polarity 0)
    Red   channel = ON  events (channel 1 / polarity 1)
    Cyan  circle  = TDTracker prediction (raw DVS-norm coords, y=0 at top)

    frame values are 0–15; divide by 15 only for display.
    """
    import cv2

    s   = _VIZ_SCALE
    bgr = np.zeros((DVS_H * s, DVS_W * s, 3), dtype=np.uint8)

    for c, channel_idx in ((1, 0), (2, 1)):   # green=OFF, red=ON
        ch = (np.clip(frame[channel_idx] / _BINAREP_MAX, 0.0, 1.0) * 255).astype(np.uint8)
        bgr[:, :, c] = cv2.resize(ch, (DVS_W * s, DVS_H * s), interpolation=cv2.INTER_NEAREST)

    if result is not None:
        (x, y), conf = result
        px = int(x * DVS_W * s)
        py = int(y * DVS_H * s)
        cv2.circle(bgr, (px, py), 10, (255, 255, 0), 2)
        cv2.circle(bgr, (px, py),  2, (255, 255, 0), -1)
        cv2.putText(bgr, f"conf {conf:.2f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return bgr


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class HybridDetectorPlugin(Detector2DPlugin):
    """
    Hybrid RITnet (RGB, ~120Hz) + TDTracker (DVS, ~1000Hz) pupil detector.

    RITnet publishes via the standard recent_events() / detect_RITnet() path.
    TDTracker runs in four background threads (dvs_thread, infer_thread,
    publish_thread, viz_thread) and publishes to "pupil.{id}.dvs".

    DVSRGBCalibrator collects (dvs_pos, rgb_pos) pairs passively and
    estimates a homography H that maps un-flipped DVS-norm → Pupil convention.
    Press F8 (or use the menu button) to force a fit from accumulated pairs.
    """

    label                    = "Hybrid RITnet+DVS"
    pupil_detection_identifier = "2d_hybrid"
    pupil_detection_method   = "hybrid ritnet+dvs"
    order                    = 0.099

    _CKPT_NAME = "best_checkpoint.pth"
    _H_NAME    = "dvs_rgb_H.npy"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        g_pool=None,
        properties=None,
        detector_2d=None,
        binarep_backend: Optional[str] = None,
    ):
        super().__init__(g_pool=g_pool, properties=properties, detector_2d=detector_2d)

        self._eye_id      = g_pool.eye_id
        self._ipc_pub_url = g_pool.ipc_pub_url
        self._get_ts      = g_pool.get_timestamp
        self._binarep_backend = _resolve_binarep_backend(binarep_backend)

        self._frame_size  = (400, 400)   # updated each RGB frame in detect_RITnet

        plugin_dir = os.path.dirname(__file__)
        user_dir = getattr(g_pool, "user_dir", plugin_dir)
        self._calibrator = DVSRGBCalibrator(
            save_path=os.path.join(user_dir, self._H_NAME)
        )

        # Rolling buffer of recent DVS results for calibration pairing.
        # Entries: (pupil_ts, dvs_norm_pos, confidence)
        # dvs_norm_pos is in DVS-native convention (un-flipped, y=0 top).
        self._dvs_buf: deque = deque(maxlen=64)
        self._dvs_buf_lock = threading.Lock()

        # Latest DVS position in Pupil convention (y=0 bottom), written by
        # the DVS thread and read by gl_display.  Raw inference — always live.
        self._display_pos: list = [None]   # Optional[Tuple[float, float]]

        # Shared publish/anchor state.  Single-element lists keep replacement
        # writes atomic enough for this plugin's producer/consumer threads.
        # State tuples: (norm_pos, confidence, source_ts, source)
        self._dvs_pub_state:        list = [None]
        self._last_good_pub_state:  list = [None]
        self._ritnet_anchor:        list = [None]
        self._anchor_offset:        list = [None]   # (dx, dy) in Pupil norm coords

        self._running = False
        self._threads: list = []

        # Only eye 0 connects to the physical DAVIS346.
        if self._eye_id != 0:
            logger.info("[Hybrid] eye1 — DVS disabled, running RITnet only.")
            return

        ckpt = os.path.join(plugin_dir, self._CKPT_NAME)
        if not os.path.exists(ckpt):
            logger.warning(f"[Hybrid] Checkpoint not found: {ckpt}. DVS disabled.")
            return

        self._ckpt_path = ckpt

        # Shared state between threads.  Single-element lists for GIL-atomic reads.
        # _latest: most recent raw TDTracker output — ((x, y), conf)
        self._latest: list = [None]
        self._infer_q = queue.Queue(maxsize=2)
        self._viz_q   = queue.Queue(maxsize=1)

        self._running = True
        logger.info(f"[Hybrid] BinarRep backend: {self._binarep_backend}")
        self._start_threads()

    def _start_threads(self):
        for name, target in (
            ("dvs_infer",   self._infer_loop),
            ("dvs_events",  self._dvs_loop),
            ("dvs_publish", self._publish_loop),
            ("dvs_viz",     self._viz_loop),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("[Hybrid] All DVS threads started.")

    # ------------------------------------------------------------------
    # Main thread: RITnet + calibration pairing
    # ------------------------------------------------------------------

    def detect_RITnet(self, frame, **kwargs):
        """Run RITnet, then pair its result with the closest DVS detection."""
        self._frame_size = (frame.width, frame.height)
        datum = super().detect_RITnet(frame, **kwargs)

        if datum is None or datum["confidence"] < RITNET_ANCHOR_CONF_THRESHOLD:
            return datum

        self._set_publish_state(
            datum["norm_pos"], datum["confidence"], datum["timestamp"], "ritnet"
        )
        self._ritnet_anchor[0] = self._last_good_pub_state[0]

        entry = self._closest_dvs(datum["timestamp"])
        if entry is not None:
            _ts, dvs_pos, dvs_conf = entry
            if dvs_conf > 0.3:
                self._calibrator.add_pair(dvs_pos, datum["norm_pos"])
                mapped = self._map_dvs_to_nir(dvs_pos)
                if mapped is not None:
                    self._update_anchor_offset(mapped, datum["norm_pos"])

        return datum

    def _closest_dvs(self, target_ts: float) -> Optional[tuple]:
        """Return the DVS buffer entry closest to target_ts, or None if too far."""
        with self._dvs_buf_lock:
            entries = tuple(self._dvs_buf)
        if not entries:
            return None
        best = min(entries, key=lambda e: abs(e[0] - target_ts))
        return best if abs(best[0] - target_ts) <= MAX_PAIR_GAP_SEC else None

    def _clip_norm_pos(self, pos) -> Optional[Tuple[float, float]]:
        if pos is None:
            return None
        x, y = float(pos[0]), float(pos[1])
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return (float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0)))

    def _make_publish_state(self, norm_pos, confidence, source_ts, source) -> Optional[tuple]:
        norm_pos = self._clip_norm_pos(norm_pos)
        if norm_pos is None:
            return None
        return (norm_pos, float(np.clip(confidence, 0.0, 1.0)), float(source_ts), source)

    def _set_publish_state(self, norm_pos, confidence, source_ts, source) -> Optional[tuple]:
        state = self._make_publish_state(norm_pos, confidence, source_ts, source)
        if state is None:
            return None
        if source == "dvs":
            self._dvs_pub_state[0] = state
        if state[1] > 0.0:
            self._last_good_pub_state[0] = state
        return state

    def _select_publish_state(self) -> Optional[tuple]:
        state = self._dvs_pub_state[0]
        if state is not None:
            _norm_pos, _conf, source_ts, _source = state
            if self._get_ts() - source_ts <= DVS_STATE_HOLD_SEC:
                return state

        anchor = self._ritnet_anchor[0]
        if anchor is not None:
            return anchor

        return self._last_good_pub_state[0]

    def _map_dvs_to_nir(self, dvs_norm) -> Optional[Tuple[float, float]]:
        """Map DVS-native norm coords into the NIR/RITnet norm_pos space."""
        if self._calibrator.is_ready:
            mapped = self._calibrator.apply(dvs_norm)
        else:
            # Temporary demo fallback: same normalized plane, y flipped into Pupil/NIR
            # convention. Jig-based extrinsics should replace this adapter later.
            mapped = (dvs_norm[0], 1.0 - dvs_norm[1])
        return self._clip_norm_pos(mapped)

    def _apply_anchor_offset(self, mapped_pos) -> Tuple[float, float]:
        offset = self._anchor_offset[0]
        if offset is None:
            return mapped_pos
        return self._clip_norm_pos((mapped_pos[0] + offset[0], mapped_pos[1] + offset[1]))

    def _update_anchor_offset(self, mapped_dvs, ritnet_pos):
        ritnet_pos = self._clip_norm_pos(ritnet_pos)
        mapped_dvs = self._clip_norm_pos(mapped_dvs)
        if ritnet_pos is None or mapped_dvs is None:
            return

        target = (
            float(np.clip(ritnet_pos[0] - mapped_dvs[0], -ANCHOR_MAX_OFFSET, ANCHOR_MAX_OFFSET)),
            float(np.clip(ritnet_pos[1] - mapped_dvs[1], -ANCHOR_MAX_OFFSET, ANCHOR_MAX_OFFSET)),
        )
        current = self._anchor_offset[0]
        if current is None:
            self._anchor_offset[0] = target
            return
        self._anchor_offset[0] = (
            ANCHOR_OFFSET_ALPHA * target[0] + (1.0 - ANCHOR_OFFSET_ALPHA) * current[0],
            ANCHOR_OFFSET_ALPHA * target[1] + (1.0 - ANCHOR_OFFSET_ALPHA) * current[1],
        )

    # ------------------------------------------------------------------
    # Inference thread: TDTracker GPU forward pass (~300–400Hz)
    # ------------------------------------------------------------------

    def _infer_loop(self):
        try:
            self._infer_loop_body()
        except Exception:
            logger.exception("[Hybrid] Infer thread crashed.")

    def _tdtracker_device_candidates(self):
        candidates = []
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for idx in (1, 0):
                if idx < count:
                    candidates.append(torch.device(f"cuda:{idx}"))
        candidates.append(torch.device("cpu"))
        return candidates

    def _load_tdtracker(self, args):
        last_error = None
        for device in self._tdtracker_device_candidates():
            try:
                net = Model(args).to(device).eval()
                net.load_state_dict(
                    torch.load(self._ckpt_path, map_location=device),
                    strict=False,
                )
                buf = torch.zeros(
                    1, SEQUENCE_LENGTH, 2, DVS_H, DVS_W,
                    dtype=torch.float32, device=device,
                )
                net(buf)   # warm-up: triggers cuDNN autotuning / JIT compilation
                logger.info(f"[Hybrid] TDTracker loaded on {device}.")
                logger.info("[Hybrid] TDTracker warm-up complete.")
                return device, net, buf
            except Exception as err:
                last_error = err
                logger.warning(f"[Hybrid] TDTracker load failed on {device}: {err}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        raise RuntimeError("TDTracker failed to load on all candidate devices") from last_error

    @torch.inference_mode()
    def _infer_loop_body(self):
        args   = types.SimpleNamespace(
            sensor_width     = DAVIS_WIDTH,
            sensor_height    = DAVIS_HEIGHT,
            spatial_factor   = 4,
            pixel_tolerances = [1, 3, 5, 10],
        )
        _device, net, buf = self._load_tdtracker(args)

        while self._running:
            try:
                seq_np = self._infer_q.get(timeout=0.1)
            except queue.Empty:
                continue

            while True:
                try:
                    seq_np = self._infer_q.get_nowait()
                except queue.Empty:
                    break

            buf.copy_(torch.from_numpy(seq_np))
            pw, ph = net(buf)
            out, prob = decode_batch_sa_simdr(pw, ph)

            x    = float(out[0, -1, 0])
            y    = float(out[0, -1, 1])
            # prob = x_prob + y_prob, each ∈ [0, 1] → total ∈ [0, 2].
            # Divide by 2 before comparing with CONF_THRESHOLD.
            conf = float((prob[0, -1] / 2.0).clamp(0.0, 1.0))

            # Atomic write: GIL guarantees single-element list assignment is safe.
            self._latest[0] = ((x, y), conf)

        logger.info("[Hybrid] Infer thread stopped.")

    # ------------------------------------------------------------------
    # DVS event loop: 1ms slices → BinarRep → infer queue → publish state
    # ------------------------------------------------------------------

    def _dvs_loop(self):
        try:
            self._dvs_loop_body()
        except Exception:
            logger.exception("[Hybrid] DVS event thread crashed.")

    def _dvs_loop_body(self):
        import sys
        import dv_processing as dv

        # Reduce CPython's GIL switch interval from 5ms to 0.5ms.
        # This lets the inference thread acquire the GIL more often during its
        # ~2.5ms forward pass, reducing end-to-end latency.
        sys.setswitchinterval(0.0005)

        capture = dv.io.camera.open()
        slicer  = dv.EventStreamSlicer()

        frame_buf: deque = deque(maxlen=SEQUENCE_LENGTH)
        ema_pos:      list = [None]   # smoothed [x, y] in DVS-native coords
        jump_blocked: list = [0]      # consecutive frames blocked by jump gate

        # ----- Clock sync: DVS µs clock → Pupil monotonic seconds -----
        # DAVIS346 runs on an independent µs counter since camera boot.
        # We anchor it to Pupil time at the first real-time slice, then
        # re-anchor every SYNC_UPDATE_INTERVAL seconds to correct drift.
        sync: list = [None]   # (dvs_us_origin, pupil_t_origin)

        def pupil_ts(dvs_us: int) -> float:
            dvs_us0, pupil_t0 = sync[0]
            return (dvs_us - dvs_us0) * 1e-6 + pupil_t0

        # ----- Staleness detection: skip USB backlog on startup -----
        # The camera buffers events while Python starts up (~1–2s of backlog).
        # We skip slices until DVS elapsed time catches up to wall-clock time.
        backlog_origin: list = [None]   # (dvs_us_start, wall_start)

        def is_stale(dvs_us: int) -> bool:
            if backlog_origin[0] is None:
                backlog_origin[0] = (dvs_us, time.monotonic())
                return False
            dvs_us0, wall_t0 = backlog_origin[0]
            return (time.monotonic() - wall_t0) - (dvs_us - dvs_us0) * 1e-6 > MAX_LAG_SEC

        rate = _RateCounter()

        def on_slice(events):
            ts = events.timestamps().astype(np.int64)
            if len(ts) == 0 or is_stale(int(ts[0])):
                return

            # Anchor / refresh clock sync
            pupil_now = self._get_ts()
            if sync[0] is None:
                sync[0] = (int(ts[0]), pupil_now)
                logger.info("[Hybrid] DVS clock anchored to Pupil timestamp.")
            elif (pupil_now - sync[0][1]) > SYNC_UPDATE_INTERVAL:
                sync[0] = (int(ts[0]), pupil_now)
                logger.debug("[Hybrid] DVS clock re-anchored (drift correction).")

            # Scale raw event coordinates to TDTracker input resolution
            coords = events.coordinates()
            xs = np.clip(
                (coords[:, 0].astype(np.float32) * (DVS_W / DAVIS_WIDTH)).astype(np.int32),
                0, DVS_W - 1,
            )
            ys = np.clip(
                (coords[:, 1].astype(np.float32) * (DVS_H / DAVIS_HEIGHT)).astype(np.int32),
                0, DVS_H - 1,
            )
            pols = events.polarities().astype(np.int64)

            try:
                frame = _to_binarep(
                    xs,
                    ys,
                    pols,
                    ts - ts[0],
                    backend=self._binarep_backend,
                )
            except ImportError as err:
                logger.warning(
                    f"[Hybrid] tonic backend unavailable ({err}); falling back to numpy."
                )
                self._binarep_backend = "numpy"
                frame = _to_binarep_numpy(xs, ys, pols, ts - ts[0])
            frame_buf.append(frame)

            # Push to viz — non-blocking, always shows latest frame
            try:
                self._viz_q.put_nowait((frame, self._latest[0]))
            except queue.Full:
                pass

            # Submit the full sliding window for inference
            if len(frame_buf) == SEQUENCE_LENGTH:
                seq = np.stack(frame_buf)[np.newaxis].astype(np.float32)   # (1, 8, 2, H, W)
                try:
                    self._infer_q.put_nowait(seq)
                except queue.Full:
                    while True:
                        try:
                            self._infer_q.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        self._infer_q.put_nowait(seq)
                    except queue.Full:
                        pass

            result = self._latest[0]
            if result is None:
                return   # no inference yet; nothing to publish or display

            (x_raw, y_raw), conf = result

            # --- EMA filter: smooth position for publish + calibration ---
            # Gate on confidence and outlier distance to avoid smoothing noise.
            # Recovery: after JUMP_RESET_CNT consecutive blocks, assume saccade and reset.
            if conf >= CONF_THRESHOLD:
                if ema_pos[0] is None:
                    ema_pos[0] = [x_raw, y_raw]
                    jump_blocked[0] = 0
                else:
                    px, py = ema_pos[0]
                    if abs(x_raw - px) <= MAX_JUMP and abs(y_raw - py) <= MAX_JUMP:
                        ema_pos[0][0] = EMA_ALPHA * x_raw + (1 - EMA_ALPHA) * px
                        ema_pos[0][1] = EMA_ALPHA * y_raw + (1 - EMA_ALPHA) * py
                        jump_blocked[0] = 0
                    else:
                        jump_blocked[0] += 1
                        if jump_blocked[0] >= JUMP_RESET_CNT:
                            # Genuine saccade — snap EMA to current position
                            ema_pos[0] = [x_raw, y_raw]
                            jump_blocked[0] = 0
                            logger.debug("[Hybrid] EMA reset: saccade detected.")

            # Smooth position for publish / calibration (fall back to raw if EMA not ready)
            smooth_x, smooth_y = ema_pos[0] if ema_pos[0] else (x_raw, y_raw)

            dvs_norm = (smooth_x, smooth_y)   # DVS-native coords — NOT flipped

            event_ts = pupil_ts(int(ts[-1]))
            with self._dvs_buf_lock:
                self._dvs_buf.append((event_ts, dvs_norm, conf))

            mapped = self._map_dvs_to_nir(dvs_norm)
            if mapped is not None:
                norm_pos = self._apply_anchor_offset(mapped)
                if norm_pos is not None:
                    self._set_publish_state(norm_pos, conf, event_ts, "dvs")
                    self._display_pos[0] = norm_pos

            hz = rate.tick()
            if hz is not None:
                logger.info(
                    f"[Hybrid] DVS slice rate: {hz:.0f} Hz | "
                    f"H_ready={self._calibrator.is_ready} | "
                    f"pairs={self._calibrator.n_pairs} | "
                    f"infer_q={self._infer_q.qsize()}"
                )

        slicer.doEveryTimeInterval(timedelta(microseconds=TIME_WINDOW_US), on_slice)

        logger.info("[Hybrid] DVS event loop running (1ms slices).")
        while self._running and capture.isRunning():
            if capture.isEventStreamAvailable():
                batch = capture.getNextEventBatch()
                if batch is not None and len(batch) > 0:
                    slicer.accept(batch)
            time.sleep(0)   # yield GIL without blocking the poll loop

        logger.info("[Hybrid] DVS event loop stopped.")

    # ------------------------------------------------------------------
    # Publish thread: 1ms precision timer → IPC
    # ------------------------------------------------------------------

    def _publish_loop(self):
        try:
            self._publish_loop_body()
        except Exception:
            logger.exception("[Hybrid] Publish thread crashed.")

    def _publish_loop_body(self):
        import zmq
        from zmq_tools import Msg_Streamer

        streamer = Msg_Streamer(zmq.Context.instance(), self._ipc_pub_url)
        eye_id   = self._eye_id
        rate     = _RateCounter()

        INTERVAL = 0.001   # 1ms target
        next_t   = time.perf_counter()

        while self._running:
            # Release the GIL for most of the 1ms interval so RITnet and the
            # inference thread aren't starved.  Only busy-wait the last ~0.1ms
            # for the sub-millisecond timing precision that sleep() can't hit.
            gap = next_t - time.perf_counter()
            if gap > 0.0002:
                time.sleep(gap - 0.0001)
            while time.perf_counter() < next_t:
                pass
            next_t += INTERVAL

            state = self._select_publish_state()
            if state is None:
                continue

            norm_pos, pub_conf, _source_ts, _source = state
            streamer.send({
                "id":         eye_id,
                "topic":      f"pupil.{eye_id}.dvs",
                "method":     "dvs tdtracker",
                "norm_pos":   norm_pos,
                "diameter":   0.0,
                "confidence": pub_conf,
                "timestamp":  self._get_ts(),   # fresh ts every send → 1000 unique entries/s
                "ellipse":    {"center": (0., 0.), "axes": (0., 0.), "angle": 0.},
            })

            hz = rate.tick()
            if hz is not None:
                logger.info(f"[Hybrid] DVS publish rate: {hz:.0f} Hz")

        logger.info("[Hybrid] Publish thread stopped.")

    # ------------------------------------------------------------------
    # Visualiser thread: live cv2 DVS window
    # ------------------------------------------------------------------

    def _viz_loop(self):
        import cv2
        WIN = "TDTracker DVS"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DVS_W * _VIZ_SCALE, DVS_H * _VIZ_SCALE)

        while self._running:
            try:
                frame, result = self._viz_q.get(timeout=0.1)
            except queue.Empty:
                cv2.waitKey(1)
                continue
            cv2.imshow(WIN, _render_dvs(frame, result))
            cv2.waitKey(1)

        cv2.destroyWindow(WIN)

    # ------------------------------------------------------------------
    # UI: menu + F8 force-fit
    # ------------------------------------------------------------------

    def init_ui(self):
        super().init_ui()
        from pyglui import ui
        self.menu.append(ui.Info_Text(
            "Press F8 (or the button below) to immediately fit the DVS↔RGB "
            "homography from all accumulated pairs."
        ))
        self.menu.append(ui.Button("Force DVS↔RGB Fit (F8)", self._force_fit))

    def on_key(self, key, scancode, action, mods):
        import glfw
        if action == glfw.PRESS and key == glfw.KEY_F8:
            self._force_fit()
            return True
        return False

    def _force_fit(self):
        ok = self._calibrator.force_fit()
        msg = f"[Hybrid] Force fit {'OK' if ok else 'FAILED'} | pairs={self._calibrator.n_pairs}"
        (logger.info if ok else logger.warning)(msg)

    # ------------------------------------------------------------------
    # GL overlay: blue ellipse (RITnet) + orange dot (DVS)
    # ------------------------------------------------------------------

    def gl_display(self):
        super().gl_display()   # RITnet blue ellipse

        pos = self._display_pos[0]
        if pos is None:
            return

        w, h         = self._frame_size
        norm_x, norm_y = pos
        # norm_pos: (0,0) = bottom-left → pixel: y must be flipped
        px = norm_x * w
        py = (1.0 - norm_y) * h
        _draw_circle_filled((px, py), size=8, color=RGBA(1.0, 0.5, 0.0, 0.9))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        super().cleanup()
