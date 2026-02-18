# -*- coding: utf-8 -*-
"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import numpy as np
import os

import glfw
from gl_utils import (
    GLFWErrorReporting,
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
)
from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

GLFWErrorReporting.set_default()

from methods import normalize
from plugin import Plugin

from . import color_scheme
from .detector_base_plugin import PupilDetectorPlugin
from .visualizer_2d import draw_pupil_outline
from pupil_detector_plugins import deepvog
from pupil_detector_plugins import edgaze
from draw_ellipse import fit_ellipse
from CheckEllipse import computeEllipseConfidence
import cv2
import torch
from pupil_detector_plugins.utils import get_predictions
from pupil_detector_plugins.models import model_dict
import time

COLOR_MAX = 255
COLOR_CAP = 256
EYE_CLASS = 1
IMAGE_MOD = 16
BBOX_EXTRA_SPACE = 20
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8
logger = logging.getLogger(__name__)


class Detector2DPlugin(PupilDetectorPlugin):
    pupil_detection_identifier = "2d"
    pupil_detection_method = "2d c++"

    label = "C++ 2d detector"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)
    order = 0.100

    @property
    def pretty_class_name(self):
        return "Pupil Detector 2D"

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.detector_2d

    def __init__(
        self,
        g_pool=None,
        properties=None,
        detector_2d: Detector2D = None,
    ):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(properties or {})

        # 1) CLAHE와 LUT 테이블을 한 번만 생성
        self.clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE))
        self.gamma_table = (COLOR_MAX * (np.linspace(0, 1, COLOR_CAP) ** 0.8)).astype(np.uint8)

        # 2) 모델 로드 (CPU)
        model_name = "densenet"
        model_path = "./best_model.pkl"
        self.device = torch.device("cpu")

        if model_name not in model_dict:
            logger.error(f"Model {model_name} not found. Valid: {list(model_dict.keys())}")
            raise ValueError("Invalid model name.")

        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} not found!")
            raise FileNotFoundError(model_path)

        state_dict = torch.load(model_path, map_location="cpu")
        self.model = model_dict[model_name]().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_init_dict(self):
        init_dict = super().get_init_dict()
        init_dict["properties"] = self.detector_2d.get_properties()
        return init_dict

    def detect(self, frame, **kwargs):
        # 기존 2D C++ detector 호출
        roi = Roi(*self.g_pool.roi.bounds)
        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None

        result = self.detector_2d.detect(
            gray_img=frame.gray,
            color_img=debug_img,
            roi=roi,
        )

        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )

        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )

        datum["ellipse"] = {
            "axes": result["ellipse"]["axes"],
            "angle": result["ellipse"]["angle"],
            "center": result["ellipse"]["center"],
        }
        return datum

    def convert_mjpeg_to_numpy(self, frame):
        try:
            img_array = np.frombuffer(frame.jpeg_buffer, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except AttributeError as e:
            raise AttributeError(f"frame 객체에서 jpeg_buffer를 찾을 수 없습니다: {e}")
        except Exception as e:
            raise RuntimeError(f"MJPEG 데이터를 numpy로 변환하는 중 오류 발생: {e}")

    def get_img(self, img: np.ndarray) -> torch.Tensor:
        """
        1) Gamma correction (0.8) using precomputed LUT
        2) CLAHE
        3) NumPy -> Torch.Tensor, Normalize([-1, +1])
        """
        # (H, W) = img.shape
        img_gamma = cv2.LUT(img, self.gamma_table)
        img_clahe = self.clahe.apply(img_gamma)

        arr = img_clahe.astype(np.float32) / 255.0          # [0,1]
        arr = (arr - 0.5) / 0.5                              # [-1,1]
        tensor = torch.from_numpy(arr).unsqueeze(0)          # (1, H, W)
        return tensor

    def find_bbox(self, img_mask: np.ndarray) -> dict:
        """
        OpenCV boundingRect을 이용해 가장 큰 컨투어의 바운딩 박스를 반환.
        img_mask: uint8 이진 마스크(0 또는 255)
        """
        contours, _ = cv2.findContours(
            img_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return {"x_min": 0, "x_max": img_mask.shape[1], "y_min": 0, "y_max": img_mask.shape[0]}

        best = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        return {"x_min": x, "x_max": x + w, "y_min": y, "y_max": y + h}

    def extract_pupil(self, predict: np.ndarray) -> np.ndarray:
        """
        predict: (H, W) 형태의 라벨 맵. 동공 라벨(값 3)만 남긴 뒤 이진 마스크 생성.
        """
        mask = np.zeros_like(predict, dtype=np.uint8)
        mask[predict == 3] = 255

        bbox = self.find_bbox(mask)
        if np.max(mask) > 0:
            norm_mask = mask.astype(np.float32) / 255.0
        else:
            norm_mask = mask.astype(np.float32)

        blank = np.zeros_like(norm_mask)
        y0, y1 = bbox["y_min"], bbox["y_max"]
        x0, x1 = bbox["x_min"], bbox["x_max"]
        blank[y0:y1, x0:x1] = norm_mask[y0:y1, x0:x1]

        blank[blank < 1.0] = 0.0
        return blank[np.newaxis, ...]  # (1, H, W)

    def detect_RITnet(self, frame, **kwargs):
        """
        RITnet으로 동공을 검출하고, Pupil Labs datum 형태로 반환.
        1) BGR → GRAY
        2) get_img() → Tensor
        3) 모델 추론
        4) get_predictions → (1, H, W) 라벨 맵
        5) 동공 라벨(3) 이진 마스크 → 컨투어+fitEllipse
        6) Pupil Labs datum 생성
        """
        if not isinstance(frame, np.ndarray):
            try:
                img = self.convert_mjpeg_to_numpy(frame)
            except ValueError as e:
                print(f"Error converting MJPEGFrame: {e}")
                return None
        else:
            img = frame

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # 1) 전처리 → Tensor (CPU)
        tensor = self.get_img(gray).unsqueeze(0)  # (1, 1, H, W)

        # 2) 추론 (CPU)
        with torch.no_grad():
            output = self.model(tensor)

        # 3) 라벨 맵 추출
        predict = get_predictions(output)           # (1, H, W)
        predict_2d = predict[0].cpu().numpy()       # (H, W)

        # 4) 동공 라벨(3)만 이진 마스크로
        pupil_mask = np.zeros_like(predict_2d, dtype=np.uint8)
        pupil_mask[predict_2d == 3] = 255

        # 5) 컨투어에서 타원 피팅
        contours, _ = cv2.findContours(
            pupil_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            result = {
                "location": (0.0, 0.0),
                "diameter": 0.0,
                "confidence": 0.0,
                "ellipse": {"axes": (0.0, 0.0), "angle": 0.0, "center": (0.0, 0.0)},
            }
        else:
            best_contour = max(contours, key=cv2.contourArea)
            if len(best_contour) < 5:
                result = {
                    "location": (0.0, 0.0),
                    "diameter": 0.0,
                    "confidence": 0.0,
                    "ellipse": {"axes": (0.0, 0.0), "angle": 0.0, "center": (0.0, 0.0)},
                }
            else:
                (cx, cy), (MA, ma), angle_deg = cv2.fitEllipse(best_contour)
                result = {
                    "location": (float(cx), float(cy)),
                    "diameter": float(MA),
                    "confidence": 1.0,
                    "ellipse": {
                        "axes": (float(MA), float(ma)),
                        "angle": float(angle_deg),
                        "center": (float(cx), float(cy)),
                    },
                }

        norm_pos = normalize(result["location"], (frame.width, frame.height), flip_y=True)
        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )
        datum["ellipse"] = {
            "axes": result["ellipse"]["axes"],
            "angle": result["ellipse"]["angle"],
            "center": result["ellipse"]["center"],
        }
        return datum

    def convert_to_builtin(self, obj):
        """
        NumPy → Python built‐in (dict/list/tuple) 변환
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_to_builtin(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_builtin(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_builtin(i) for i in obj)
        else:
            return obj

    def unproject_single_observation(self, prediction, mask=None, threshold=0.5):
        ellipse_info = fit_ellipse(prediction, mask=mask)
        ellipse_confidence = 0

        if ellipse_info is not None:
            rr, cc, centre, w, h, radian, ell = ellipse_info
            ellipse_confidence = computeEllipseConfidence(prediction, centre, w, h, radian)
            return {
                "ellipse": {"center": (float(centre[0]), float(centre[1])),
                            "axes":   (float(w), float(h)),
                            "angle":  float(np.degrees(radian))},
                "diameter":  float(h),
                "location":  (float(centre[0]), float(centre[1])),
                "confidence": float(ellipse_confidence),
            }
        else:
            return {
                "ellipse":   {"center": (0.0, 0.0), "axes": (0.0, 0.0), "angle": 0.0},
                "diameter":  0.0,
                "location":  (0.0, 0.0),
                "confidence": 0.0,
            }

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "
            + "Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "
            + "Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Slider(
                "intensity_range",
                self.pupil_detector_properties,
                label="Pupil intensity range",
                min=0,
                max=60,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_min",
                self.pupil_detector_properties,
                label="Pupil min",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_max",
                self.pupil_detector_properties,
                label="Pupil max",
                min=50,
                max=400,
                step=1,
            )
        )
        info = ui.Info_Text(
            "When using Neon in bright light, increasing the Canny Threshold can "
            "help reduce the effect of reflections in the eye image and improve pupil "
            "detection. The default value is 160."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Slider(
                "canny_treshold",
                self.pupil_detector_properties,
                label="Canny Threshold",
                min=0,
                max=1000,
                step=1,
            )
        )
        self.menu.append(ui.Info_Text("Color Legend"))
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_2D.as_float, "2D pupil ellipse")
        )

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(
                self._recent_detection_result,
                color_rgb=color_scheme.PUPIL_ELLIPSE_2D.as_float,
            )

    def on_resolution_change(self, old_size, new_size):
        properties = self.pupil_detector.get_properties()
        properties["pupil_size_max"] *= new_size[0] / old_size[0]
        properties["pupil_size_min"] *= new_size[0] / old_size[0]
        self.pupil_detector.update_properties(properties)
