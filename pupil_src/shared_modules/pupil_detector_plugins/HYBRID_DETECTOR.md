# Hybrid RITnet + DVS Detector

This document describes the current hybrid pupil detector implementation.

The intended role is simple:

```text
RITnet / NIR eye camera:
  accurate absolute pupil anchor at eye-camera frame rate

DVS / TDTracker:
  fast auxiliary pupil position source used to fill a 1000 Hz output stream

Hybrid publisher:
  emits a fresh-timestamp pupil stream every 1 ms using latest DVS, RITnet
  anchor, or last-good fallback
```

The DVS path is not intended to estimate gaze vectors yet. It only assists
high-rate pupil-center output for eye 0.

## Default Loading

`available_detector_plugins()` currently returns `HybridDetectorPlugin` instead
of the standalone `Detector2DPlugin`.

This means a normal Capture launch will load the hybrid detector by default.
The standalone 2D detector class still exists because the hybrid plugin reuses
its RITnet implementation through inheritance, but it is not loaded as a
separate detector.

If `Pye3DPlugin` is available, it may still be appended by the normal detector
registration path. The current hybrid DVS path does not require Pye3D.

## Output Topics

The inherited RITnet path emits normal detector output using the hybrid detector
identifier:

```text
pupil.<eye_id>.2d_hybrid
```

The DVS high-rate auxiliary stream emits:

```text
pupil.<eye_id>.dvs
```

Only eye 0 starts the DAVIS/DVS threads. Eye 1 runs RITnet-only behavior and
does not open the physical DVS camera.

## Thread Model

The hybrid plugin uses the normal eye-process main thread plus DVS background
threads:

```text
main thread:
  run RITnet per NIR frame
  update RITnet anchor state
  pair nearby DVS samples for temporary homography calibration

dvs_events thread:
  read DAVIS event batches
  slice events into 1 ms windows
  build BinarRep frames with the selected backend, numpy by default
  update smoothed DVS state

dvs_infer thread:
  run TDTracker on latest sequence only
  prefer cuda:1, then cuda:0, then CPU

dvs_publish thread:
  run a 1 ms timer
  publish latest selected pupil state with a fresh timestamp

dvs_viz thread:
  optional OpenCV DVS preview window
```

## 1000 Hz Contract

The current contract is output-rate oriented:

```text
The publisher tries to emit one datum every 1 ms.
Each emitted datum gets a fresh `timestamp`.
The position may be the latest DVS state, a held RITnet anchor, or a held
last-good state.
```

This does not mean TDTracker produces a new independent inference every 1 ms.
Inference is latest-only: stale queued sequences are discarded so latency stays
low. If inference is slower than the publish loop, the publisher holds the most
recent usable state.

This is intentional for the current demo stage.

## Position Selection

The publisher chooses a state in this priority order:

```text
1. recent DVS publish state, if not older than DVS_STATE_HOLD_SEC
2. latest RITnet anchor
3. last-good state
```

The DVS state is produced as:

```text
DVS model output
  -> DVS normalized coordinate, y=0 at top
  -> _map_dvs_to_nir()
  -> anchor offset correction from RITnet
  -> clipped NIR/Pupil normalized coordinate
```

The RITnet anchor state is produced from `detect_RITnet()` when confidence is
above `RITNET_ANCHOR_CONF_THRESHOLD`.

## RITnet Anchor Correction

RITnet is treated as the absolute position source. DVS is treated as a fast
auxiliary source.

When a confident RITnet result has a nearby DVS sample, the plugin updates an
offset:

```text
target_offset = ritnet_norm_pos - mapped_dvs_norm_pos
anchor_offset = EMA(anchor_offset, target_offset)
```

The DVS output is then corrected:

```text
published_dvs_pos = mapped_dvs_pos + anchor_offset
```

This keeps DVS motion fast while slowly pulling it back toward the RITnet
absolute coordinate frame.

## DVS Noise Handling

The DVS path applies lightweight smoothing and outlier handling:

```text
CONF_THRESHOLD:
  only update the DVS EMA when TDTracker confidence is high enough

MAX_JUMP:
  reject isolated large jumps

JUMP_RESET_CNT:
  after repeated large jumps, treat the movement as a real saccade and reset
  the EMA to the new position
```

This is deliberately simple. The current goal is stable-looking high-rate output,
not final precision.

## BinarRep Backend: NumPy Or tonic

The hybrid live path can use either backend:

```text
numpy:
  low-overhead local implementation
  default backend

tonic:
  reference-style tonic.transforms implementation
  useful when you want to compare against the training/preprocessing path
```

Selection options:

```python
# In main.py, before eye processes start, change the setdefault value:
os.environ.setdefault("PUPIL_HYBRID_BINAREP_BACKEND", "tonic")

# Or when manually constructing/loading the plugin:
HybridDetectorPlugin(..., binarep_backend="tonic")

# Or directly in detector_2d_hybrid_plugin.py:
BINAREP_BACKEND_DEFAULT = "tonic"
```

If `tonic` is selected but unavailable, the plugin logs a warning and falls back
to `numpy`.

The default remains `numpy` because it avoids transform-object overhead in the
1 ms path and does not require `tonic` at runtime.

The older standalone `dvs_detector_plugin.py` still imports `tonic.transforms`,
but that plugin is not the current hybrid production path.

## Coordinate Conventions

DVS native/model coordinates use top-left origin:

```text
dvs_norm = (x, y), y=0 at top
```

Pupil/NIR normalized coordinates use bottom-left origin:

```text
norm_pos = (x, y), y=0 at bottom
```

`_map_dvs_to_nir()` is the adapter boundary. Currently it does:

```text
if homography is ready:
  use DVSRGBCalibrator homography
else:
  use temporary normalized fallback: (x, 1 - y)
```

Later, this function should be replaced with the fixed jig/extrinsic projection:

```text
DVS normalized point
  -> DVS ray / camera model
  -> fixed extrinsic transform
  -> NIR image projection
  -> NIR/Pupil norm_pos
```

## Temporary Homography

`DVSRGBCalibrator` passively collects paired observations:

```text
(DVS normalized position, RITnet normalized position)
```

It estimates a homography with RANSAC and saves it to the user settings
directory. It validates loaded/computed homographies and logs warnings instead
of crashing when save/load fails.

The homography is temporary and should not be treated as the final calibration
model. The final design should use fixed extrinsic parameters after the jig is
installed.

## Failure Behavior

Expected fallback behavior:

```text
No DVS camera:
  DVS event thread fails/logs; RITnet path can still run

No cuda:1:
  try cuda:0

No CUDA:
  try CPU

No homography:
  use normalized y-flip fallback

No fresh DVS state:
  publish RITnet anchor or last-good state
```

The remaining practical risk is resource load: the publisher uses a 1 ms timing
loop with a short busy-wait, and actual timing depends on OS scheduling, CPU/GPU
load, and camera/event throughput.

## Current Non-Goals

The current hybrid detector does not attempt to:

```text
estimate gaze vectors from DVS
replace gaze mapping
run binocular DVS tracking
produce a 3D eye model
guarantee a new TDTracker inference every 1 ms
finalize DVS/NIR calibration
```

## Important Parameters

The main tuning constants are defined in `detector_2d_hybrid_plugin.py`:

```text
TIME_WINDOW_US:
  DVS event slicing interval, currently 1000 us

SEQUENCE_LENGTH:
  TDTracker temporal window length

CONF_THRESHOLD:
  DVS confidence gate for EMA updates

EMA_ALPHA:
  DVS smoothing factor

MAX_JUMP / JUMP_RESET_CNT:
  outlier gate and saccade recovery

RITNET_ANCHOR_CONF_THRESHOLD:
  minimum RITnet confidence for anchor updates

ANCHOR_OFFSET_ALPHA:
  EMA rate for RITnet-based offset correction

DVS_STATE_HOLD_SEC:
  how long recent DVS state is preferred before falling back to anchor/last-good
```

## Recommended Next Steps

1. Measure actual `pupil.0.dvs` publish rate with the DAVIS connected.
2. Confirm TDTracker device selection on the target GPU machine.
3. Decide whether to disable Pye3D default loading for the current experiment.
4. Tune `CONF_THRESHOLD`, `EMA_ALPHA`, `MAX_JUMP`, and `DVS_STATE_HOLD_SEC` using
   real eye movement data.
5. Replace `_map_dvs_to_nir()` with fixed jig/extrinsic projection when the
   hardware installation is fixed.
