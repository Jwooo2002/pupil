import time
import zmq
import logging
import os

import numpy as np
import cv2
import torch
import torchvision

import glfw
from gl_utils import (
    GLFWErrorReporting,
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
)
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

from pupil_detectors import Detector2D, Roi
from methods import normalize
from plugin import Plugin
from . import color_scheme
from .detector_base_plugin import PupilDetectorPlugin
from .visualizer_2d import draw_pupil_outline
from pupil_detector_plugins.utils import get_predictions
from pupil_detector_plugins.models import model_dict

GLFWErrorReporting.set_default()
logger = logging.getLogger(__name__)

COLOR_CAP      = 256
CLIP_LIMIT     = 1.5
TILE_GRID_SIZE = 8


class Detector2DPlugin(PupilDetectorPlugin):
    pupil_detection_identifier = "2d"
    pupil_detection_method     = "2d c++"

    label     = "2D + DVS Mixer"
    icon_font = "pupil_icons"
    icon_chr  = chr(0xEC18)
    order     = 0.100

    def update(self, frame, timestamp, *args, **kwargs):
        # 매 프레임 DVS 구독 시도
        self._try_recv_dvs()
        super().update(frame, timestamp, *args, **kwargs)

    @property
    def pretty_class_name(self):
        return "Pupil Detector 2D + DVS"

    @property
    def pupil_detector(self):
        return self.detector_2d

    def __init__(self, g_pool=None, properties=None, detector_2d=None):
        super().__init__(g_pool=g_pool)

        # 1) 기존 C++ 2D detector 초기화
        self.detector_2d = detector_2d or Detector2D(properties or {})

        # 2) RITnet 모델 초기화
        model_name = "densenet"
        model_path = "./best_model.pkl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name not in model_dict:
            raise ValueError(f"Invalid model name: {model_name}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self.model = model_dict[model_name]().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
        self.clahe = cv2.createCLAHE(
            clipLimit=CLIP_LIMIT,
            tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE),
        )

        # 3) ZMQ SUB 초기화 (DVS 파이프라인으로부터 좌표 수신)
        ctx = zmq.Context()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.connect("tcp://localhost:6000")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        time.sleep(0.01)  # 연결 안정화
        self.latest_dvs = None  # (norm_x, norm_y, ts)

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu.append(ui.Info_Text(
            "▶ 1000 Hz DVS 좌표를 우선 사용합니다.\n"
            "▶ 120 fps RITnet/C++ 결과가 준비되면 덮어씁니다."
        ))

    def _try_recv_dvs(self):
        """Non-blocking으로 최신 DVS 좌표 업데이트."""
        try:
            msg = self.sub.recv_json(flags=zmq.NOBLOCK)
            raw_x = msg["x"]
            raw_y = msg["x"]
            ts = msg.get("ts", 0) / 1e6
            self.latest_dvs = (raw_x, raw_y, ts)
        except zmq.Again:
            pass

    def detect(self, frame, **kwargs):
        # 1) DVS에서 최신 좌표 받아오기
        self._try_recv_dvs()

        # 2) RITnet 기반 검출
        datum = self.detect_RITnet(frame, **kwargs)
        if datum is None:
            # 검출 실패 시 빈 datum 보장
            datum = self.create_pupil_datum(
                norm_pos=(0.0, 0.0),
                diameter=0.0,
                confidence=0.0,
                timestamp=frame.timestamp,
            )
            datum["ellipse"] = {"axes": (0.0,0.0), "angle": 0.0, "center": (0.0,0.0)}

        # 3) DVS 좌표가 있으면 덮어쓰기
        if self.latest_dvs is not None:
            raw_x, raw_y, ts = self.latest_dvs
            x_n = raw_x / frame.width
            y_n = raw_y / frame.height
            cx, cy = raw_x, raw_y

            datum["norm_pos"]  = (x_n, y_n)
            datum["timestamp"] = ts
            datum["ellipse"]["center"] = (cx, cy)
            datum["confidence"] = max(datum["confidence"], 0.5)

        return datum

    def detect_RITnet(self, frame, **kwargs):

        self._try_recv_dvs()
        if self.latest_dvs is not None:
            print(f"[DVS SUB→RIT] latest_dvs={self.latest_dvs}")
        """
        RITnet으로 동공을 검출하고 Pupil Labs datum 형식으로 반환합니다.
        """
        # 1) 프레임 → BGR numpy
        if isinstance(frame, np.ndarray):
            img = frame
        else:
            try:
                img = self.convert_mjpeg_to_numpy(frame)
            except Exception as e:
                logger.error(f"MJPEG → numpy 변환 실패: {e}")
                return None

        # 2) 그레이스케일 & CLAHE 적용
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)

        # 3) RITnet 전처리 및 추론
        img_tensor = self.get_img(gray)                              # [1,H,W]
        data       = img_tensor.unsqueeze(0).to(self.device)          # [1,1,H,W]
        with torch.no_grad():
            output = self.model(data)                                 # [1,C,H,W]

        predict    = get_predictions(output)                          # [1,H,W]
        predict_2d = predict[0].cpu().numpy()                         # [H,W]

        # 4) 동공 라벨 → 이진 마스크 → 컨투어 → fitEllipse
        pupil_mask = np.zeros_like(predict_2d, dtype=np.uint8)
        pupil_mask[predict_2d == 3] = 255

        contours, _ = cv2.findContours(
            pupil_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            result = {
                "location":   (0.0, 0.0),
                "diameter":   0.0,
                "confidence": 0.0,
                "ellipse": {
                    "axes":   (0.0, 0.0),
                    "angle":  0.0,
                    "center": (0.0, 0.0),
                },
            }
        else:
            best = max(contours, key=cv2.contourArea)
            if len(best) < 5:
                result = {
                    "location":   (0.0, 0.0),
                    "diameter":   0.0,
                    "confidence": 0.0,
                    "ellipse": {
                        "axes":   (0.0, 0.0),
                        "angle":  0.0,
                        "center": (0.0, 0.0),
                    },
                }
            else:
                (cx, cy), (MA, ma), ang = cv2.fitEllipse(best)
                result = {
                    "location":   (float(cx), float(cy)),
                    "diameter":   float(MA),
                    "confidence": 1.0,
                    "ellipse": {
                        "axes":   (float(MA), float(ma)),
                        "angle":  float(ang),
                        "center": (float(cx), float(cy)),
                    },
                }

        # 5) Pupil Labs datum 생성
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
            "axes":   result["ellipse"]["axes"],
            "angle":  result["ellipse"]["angle"],
            "center": result["ellipse"]["center"],
        }
        return datum

    def convert_mjpeg_to_numpy(self, frame):
        arr = np.frombuffer(frame.jpeg_buffer, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def get_img(self, img: np.ndarray) -> torch.Tensor:
        """
        Gamma(0.8) → CLAHE → ToTensor → Normalize → Tensor 반환
        """
        table      = float(COLOR_CAP - 1) * (np.linspace(0,1,COLOR_CAP) ** 0.8)
        img_gamma  = cv2.LUT(img, table.astype(np.uint8))
        img_clahe  = self.clahe.apply(img_gamma)
        pil_img    = torchvision.transforms.functional.to_pil_image(img_clahe)
        tensor_img = self.transform(pil_img)
        return tensor_img

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(
                self._recent_detection_result,
                color_rgb=color_scheme.PUPIL_ELLIPSE_2D.as_float,
            )

    def on_resolution_change(self, old_size, new_size):
        props = self.pupil_detector.get_properties()
        scale = new_size[0] / old_size[0]
        props["pupil_size_max"] *= scale
        props["pupil_size_min"] *= scale
        self.pupil_detector.update_properties(props)
