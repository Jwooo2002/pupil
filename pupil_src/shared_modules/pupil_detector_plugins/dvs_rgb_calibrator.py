"""
DVS-RGB Homography Calibrator

Passively collects (dvs_pos, rgb_pos) pairs while the detector runs normally.
No special calibration procedure needed — pairs accumulate as the eye moves.
H is updated periodically and saved to disk, so it survives restarts.

DVS camera drift is handled via a rolling window: H always reflects
the most recent MAX_PAIRS observations.
"""
import logging
import os

import cv2
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

MIN_PAIRS = 20          # minimum pairs before H can be estimated
MAX_PAIRS = 200         # rolling window — handles slow camera drift
UPDATE_INTERVAL = 30    # recompute H every N new pairs
RANSAC_THRESHOLD = 0.02 # reprojection error threshold (normalized coords)


def _is_valid_homography(H):
    H = np.asarray(H)
    return H.shape == (3, 3) and np.all(np.isfinite(H))


class DVSRGBCalibrator:
    """
    Online homography estimator: DVS coordinate space → RGB coordinate space.

    Usage:
        calibrator = DVSRGBCalibrator(save_path="dvs_rgb_H.npy")

        # each time a paired observation is available:
        calibrator.add_pair(dvs_norm_pos, rgb_norm_pos)

        # apply to DVS positions:
        rgb_pos = calibrator.apply(dvs_norm_pos)  # None if H not ready yet
    """

    def __init__(self, save_path=None):
        self._pairs = deque(maxlen=MAX_PAIRS)
        self._H = None
        self._pairs_since_update = 0
        self._save_path = save_path

        if save_path and os.path.exists(save_path):
            self._load(save_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_pair(self, dvs_pos, rgb_pos):
        """
        Add a paired observation.
        Both dvs_pos and rgb_pos are (x, y) normalized to [0, 1].
        H is recomputed every UPDATE_INTERVAL new pairs.
        """
        self._pairs.append((dvs_pos, rgb_pos))
        self._pairs_since_update += 1

        if (len(self._pairs) >= MIN_PAIRS
                and self._pairs_since_update >= UPDATE_INTERVAL):
            self._update_H()
            self._pairs_since_update = 0

    def apply(self, dvs_pos):
        """
        Transform dvs_pos → RGB space using current H.
        Returns None if H has not been estimated yet.
        """
        if self._H is None:
            return None
        pt = np.array([[[dvs_pos[0], dvs_pos[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        x, y = float(out[0, 0, 0]), float(out[0, 0, 1])
        return (float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0)))

    def force_fit(self) -> bool:
        """
        Immediately compute H from all accumulated pairs (bypasses UPDATE_INTERVAL).
        Returns True if H was updated, False if not enough pairs.
        """
        if len(self._pairs) < 4:
            logger.warning(
                f"[DVSRGBCalibrator] force_fit: need ≥4 pairs, have {len(self._pairs)}"
            )
            return False
        updated = self._update_H()
        self._pairs_since_update = 0
        return updated

    @property
    def is_ready(self):
        return self._H is not None

    @property
    def n_pairs(self):
        return len(self._pairs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_H(self):
        dvs_pts = np.array([p[0] for p in self._pairs], dtype=np.float32)
        rgb_pts = np.array([p[1] for p in self._pairs], dtype=np.float32)

        H, mask = cv2.findHomography(
            dvs_pts, rgb_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
        )
        if H is None:
            logger.warning("[DVSRGBCalibrator] findHomography failed — not enough spread?")
            return False
        if not _is_valid_homography(H):
            logger.warning("[DVSRGBCalibrator] findHomography returned invalid H")
            return False

        inlier_ratio = mask.sum() / len(mask) if mask is not None else 0.0
        logger.debug(
            f"[DVSRGBCalibrator] H updated | pairs={len(self._pairs)} "
            f"inliers={inlier_ratio:.1%}"
        )
        self._H = H

        if self._save_path:
            try:
                np.save(self._save_path, H)
            except Exception as e:
                logger.warning(f"[DVSRGBCalibrator] Failed to save H: {e}")
        return True

    def _load(self, path):
        try:
            H = np.load(path)
            if not _is_valid_homography(H):
                logger.warning(f"[DVSRGBCalibrator] Ignoring invalid H from {path}")
                self._H = None
                return
            self._H = H
            logger.info(f"[DVSRGBCalibrator] Loaded H from {path}")
        except Exception as e:
            logger.warning(f"[DVSRGBCalibrator] Failed to load H: {e}")
            self._H = None
