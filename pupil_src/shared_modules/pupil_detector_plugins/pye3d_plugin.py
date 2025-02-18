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
import os
import pye3d
from methods import normalize
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
from pyglui import ui
from version_utils import parse_version

from . import color_scheme
from .detector_base_plugin import PupilDetectorPlugin
from .visualizer_2d import draw_ellipse, draw_eyeball_outline, draw_pupil_outline
from .visualizer_pye3d import Eye_Visualizer
import numpy as np
import cv2
import torch
import PIL
from pupil_detector_plugins.utils import get_predictions
from pupil_detector_plugins.models import model_dict
import torchvision

COLOR_MAX = 255
COLOR_CAP = 256
EYE_CLASS = 1
IMAGE_MOD = 16
BBOX_EXTRA_SPACE = 20
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8
EYE_CLASS = 1

logger = logging.getLogger(__name__)

version_installed = parse_version(getattr(pye3d, "__version__", "0.0.1"))
version_supported = parse_version("0.3")

if not version_installed.release[:2] == version_installed.release[:2]:
    logger.info(
        f"Requires pye3d version {version_supported} "
        f"(Installed: {version_installed})"
    )
    raise ImportError("Unsupported version found")


class Pye3DPlugin(PupilDetectorPlugin):
    pupil_detection_identifier = "3d"
    # pupil_detection_method implemented as variable

    label = "Pye3D"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC19)
    order = 0.101

    @property
    def pupil_detector(self):
        return self.detector

    def __init__(self, g_pool=None, **kwargs):
        super().__init__(g_pool=g_pool)
        self.camera = CameraModel(
            focal_length=self.g_pool.capture.intrinsics.focal_length,
            resolution=self.g_pool.capture.intrinsics.resolution,
        )
        async_apps = ("capture", "service")
        mode = (
            DetectorMode.asynchronous
            if g_pool.app in async_apps
            else DetectorMode.blocking
        )
        logger.debug(f"Running {mode.name} in {g_pool.app}")
        self.detector = Detector3D(camera=self.camera, long_term_mode=mode, **kwargs)

        method_suffix = {
            DetectorMode.asynchronous: "real-time",
            DetectorMode.blocking: "post-hoc",
        }
        self.pupil_detection_method = f"pye3d {pye3d.__version__} {method_suffix[mode]}"

        self.debugVisualizer3D = Eye_Visualizer(self.g_pool, self.camera.focal_length)
        self.__debug_window_button = None

        model_name = "densenet"
        model_path = "./best_model.pkl"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device_str)

        # 3) 모델 로드
        #    (model_dict, get_predictions 등은 RITnet 예제에서 import 했다고 가정)
        if model_name not in model_dict:
            logger.error(f"Model {model_name} not found. Valid: {list(model_dict.keys())}")
            raise ValueError("Invalid model name.")

        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} not found!")
            raise FileNotFoundError(model_path)

        self.model = model_dict[model_name].to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE)
        )

    def get_init_dict(self):
        init_dict = super().get_init_dict()
        return init_dict

    def _process_camera_changes(self):
        camera = CameraModel(
            focal_length=self.g_pool.capture.intrinsics.focal_length,
            resolution=self.g_pool.capture.intrinsics.resolution,
        )
        if self.camera == camera:
            return

        logger.debug(f"Camera model change detected: {camera}. Resetting 3D detector.")
        self.camera = camera
        self.detector.reset_camera(self.camera)

        # Debug window also depends on focal_length, need to replace it with a new
        # instance. Make sure debug window is closed at this point or we leak the opengl
        # window.
        debug_window_was_opened = self.is_debug_window_open
        self.debug_window_close()
        self.debugVisualizer3D = Eye_Visualizer(self.g_pool, self.camera.focal_length)
        if debug_window_was_opened:
            self.debug_window_open()

    def on_resolution_change(self, old_size, new_size):
        # TODO: the logic for old 2D/3D resetting does not fit here anymore, but was
        # included in the PupilDetectorPlugin base class. This needs some cleaning up.
        pass

    # def detect(self, frame, **kwargs):
    #     self._process_camera_changes()
    #
    #     previous_detection_results = kwargs.get("previous_detection_results", [])
    #     for datum in previous_detection_results:
    #         if datum.get("method", "") == "2d c++":
    #             datum_2d = datum
    #             break
    #     else:
    #         logger.warning(
    #             "Required 2d pupil detection input not available. "
    #             "Returning default pye3d datum."
    #         )
    #         return self.create_pupil_datum(
    #             norm_pos=[0.5, 0.5],
    #             diameter=0.0,
    #             confidence=0.0,
    #             timestamp=frame.timestamp,
    #         )
    #
    #     result = self.detector.update_and_detect(
    #         datum_2d, frame.gray, debug=self.is_debug_window_open
    #     )
    #
    #     norm_pos = normalize(
    #         result["location"], (frame.width, frame.height), flip_y=True
    #     )
    #     template = self.create_pupil_datum(
    #         norm_pos=norm_pos,
    #         diameter=result["diameter"],
    #         confidence=result["confidence"],
    #         timestamp=frame.timestamp,
    #     )
    #     template.update(result)
    #
    #     return template

    def get_img(self, img: np.ndarray) -> torch.Tensor:
        """
        1) Gamma correction (0.8)
        2) CLAHE
        3) PIL 변환 -> transforms.ToTensor() + Normalize([0.5],[0.5])
        4) 바로 텐서로 리턴
        """
        # (H, W) = img.shape[:2]  # 필요시 사용

        # 1) gamma correction
        table = float(COLOR_MAX) * (np.linspace(0, 1, COLOR_CAP) ** 0.8)
        img_gamma = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))

        # 2) CLAHE
        img_clahe = self.clahe.apply(img_gamma)

        # 3) PIL 변환
        pil_img = PIL.Image.fromarray(img_clahe)

        # 4) ToTensor + Normalize([0.5],[0.5])
        #   (self.transform이 이미 transforms.Compose([...])로 정의되어 있다고 가정)
        tensor_img = self.transform(pil_img)
        # tensor_img: shape [C, H, W], dtype=torch.float32, 범위 ~ [-1,1]

        return tensor_img

    def convert_mjpeg_to_numpy(self, frame):
        try:
            # frame.jpeg_buffer를 numpy 배열로 변환
            img_array = np.frombuffer(frame.jpeg_buffer, dtype=np.uint8)
            # OpenCV로 MJPEG 디코딩
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except AttributeError as e:
            raise AttributeError(f"frame 객체에서 jpeg_buffer를 찾을 수 없습니다: {e}")
        except Exception as e:
            raise RuntimeError(f"MJPEG 데이터를 numpy로 변환하는 중 오류 발생: {e}")

    def convert_to_builtin(self, obj):
        """
        재귀적으로 numpy.ndarray를 기본 Python list로 변환하는 함수.
        dict, list, tuple 내에 있는 numpy 배열도 변환합니다.
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

    def detect_RITnet(self, frame, **kwargs):
        """
        RITnet 기반 2D 동공 검출을 수행하여,
        기존 pye3d_plugin의 detect()와 동일한 출력 포맷(pupil datum)을 반환합니다.
        이전 detection 결과가 있다면, 그 중 "2d ritnet" 결과를 우선 사용합니다.
        """
        # 1) 카메라 상태 처리
        self._process_camera_changes()

        # 2) previous_detection_results에서 "2d ritnet" 결과가 있는지 확인
        previous_detection_results = kwargs.get("previous_detection_results", [])
        datum_2d = None
        for datum in previous_detection_results:
            if datum.get("method", "") == "2d c++":
                datum_2d = datum
                break

        # 3) 만약 없다면 RITnet 기반 2D 검출 수행
        # if datum_2d is None:
        #     datum_2d = self._perform_ritnet_2d(frame)
        #     if datum_2d is None:
        #         logger.warning(
        #             "RITnet-based 2D detection failed. Returning default pye3d datum."
        #         )
        #         return self.create_pupil_datum(
        #             norm_pos=[0.5, 0.5],
        #             diameter=0.0,
        #             confidence=0.0,
        #             timestamp=frame.timestamp,
        #         )
        #         # gaze mapping에서 요구하는 추가 키들을 기본값으로 설정
        #         default_datum["model_confidence"] = 0.0
        #         default_datum["projected_sphere"] = {"axes": (0.0, 0.0), "angle": 0.0, "center": (0.0, 0.0)}
        #         default_datum["sphere"] = {"axes": (0.0, 0.0), "angle": 0.0, "center": (0.0, 0.0)}
        #         default_datum["circle_3d"] = {"normal": (0.0, 0.0, 1.0)}
        #         return default_datum

        else:
            logger.warning(
                        "Required 2d pupil detection input not available. "
                        "Returning default pye3d datum."
                    )
            return self.create_pupil_datum(
                        norm_pos=[0.5, 0.5],
                        diameter=0.0,
                        confidence=0.0,
                        timestamp=frame.timestamp,
                    )


        # 4) 3D 업데이트: 2D datum와 frame.gray를 이용해 3D 모델 업데이트 및 결과 추정
        result = self.detector.update_and_detect(
            datum_2d, frame.gray, debug=self.is_debug_window_open
        )

        # 5) 결과 datum 생성: normalize, create_pupil_datum, template 업데이트
        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )
        template = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )
        template.update(result)

        return template

    def _perform_ritnet_2d(self, frame):
        """
        RITnet 기반 2D 동공 검출을 수행하고,
        Pupil Labs의 2D datum 형식(dict)을 반환하는 헬퍼 함수입니다.
        """
        # (A) frame이 np.ndarray가 아니라면 변환
        if not isinstance(frame, np.ndarray):
            try:
                img = self.convert_mjpeg_to_numpy(frame)
            except ValueError as e:
                logger.error(f"Error converting MJPEGFrame: {e}")
                return None

        # (B) 그레이스케일 변환 및 uint8 캐스팅
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)

        # (C) 전처리: 감마 보정 + CLAHE + ToTensor + Normalize
        #     self.get_img()는 [1, H, W] 텐서를 반환한다고 가정합니다.
        img_tensor = self.get_img(gray)
        data = img_tensor.unsqueeze(0).to(self.device)  # shape=[1,1,H,W]

        # (D) RITnet 추론
        with torch.no_grad():
            output = self.model(data)
        predict = get_predictions(output)  # shape=[1,H,W]
        predict_2d = predict[0].cpu().numpy()  # shape=[H,W], 정수 라벨 (예: 0..3)

        # (E) 동공 라벨 추출 (여기서는 동공이 '3'로 가정; 필요 시 '1'로 변경)
        pupil_mask = np.zeros_like(predict_2d, dtype=np.uint8)
        pupil_mask[predict_2d == 3] = 255

        # (F) Contour 찾기 및 타원 피팅
        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        best_contour = max(contours, key=cv2.contourArea)
        if len(best_contour) < 5:
            return None
        ellipse = cv2.fitEllipse(best_contour)  # ((cx,cy), (MA,ma), angle_deg)
        (cx, cy), (MA, ma), angle_deg = ellipse

        # (G) 임시 confidence 설정 (추후 정교한 계산 가능)
        conf_val = 1.0

        # (H) RITnet 기반 2D datum 생성
        datum_2d = {
            "method": "2d c++",
            "location": (float(cx), float(cy)),
            "diameter": float(MA),
            "confidence": conf_val,
            "timestamp": frame.timestamp,
            "ellipse": {
                "axes": (float(MA), float(ma)),
                "angle": float(angle_deg),
                "center": (float(cx), float(cy)),
            },
        }
        return datum_2d

    def on_notify(self, notification):
        super().on_notify(notification)

        subject = notification["subject"]
        if subject == "pupil_detector.3d.reset_model":
            if "id" not in notification:
                # simply apply to all eye processes
                self.reset_model()
            elif notification["id"] == self.g_pool.eye_id:
                # filter for specific eye processes
                self.reset_model()

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Pye3D Detector"

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name

        help_text = (
            f"pye3d {pye3d.__version__} - a model-based 3d pupil detector with corneal "
            "refraction correction. Read more about the detector in our docs website."
        )
        self.menu.append(ui.Info_Text(help_text))
        self.menu.append(ui.Button("Reset 3D model", self.reset_model))
        self.__debug_window_button = ui.Button(
            self.__debug_window_button_label, self.debug_window_toggle
        )

        help_text = (
            "The 3d model automatically updates in the background. Freeze the model to "
            "turn off automatic model updates. Refer to the docs website for details. "
        )
        self.menu.append(ui.Info_Text(help_text))
        self.menu.append(
            ui.Switch("is_long_term_model_frozen", self.detector, label="Freeze model")
        )
        self.menu.append(self.__debug_window_button)
        self.menu.append(ui.Info_Text("Color Legend - Default"))
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_3D.as_float, "3D pupil ellipse")
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_IN.as_float,
                "Long-term model outline (within bounds)",
            )
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_OUT.as_float,
                "Long-term model outline (out-of-bounds)",
            )
        )
        self.menu.append(ui.Info_Text("Color Legend - Debug"))
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_SHORT_TERM_DEBUG.as_float,
                "Short-term model outline",
            )
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_ULTRA_LONG_TERM_DEBUG.as_float,
                "Ultra-long-term model outline",
            )
        )

    def gl_display(self):
        self.debug_window_update()
        result = self._recent_detection_result

        if result is not None:
            # normal eyeball drawing
            draw_eyeball_outline(result)

            if self.is_debug_window_open and "debug_info" in result:
                # debug eyeball drawing
                debug_info = result["debug_info"]
                draw_ellipse(
                    ellipse=debug_info["projected_ultra_long_term"],
                    rgba=color_scheme.EYE_MODEL_OUTLINE_ULTRA_LONG_TERM_DEBUG.as_float,
                    thickness=2,
                )
                draw_ellipse(
                    ellipse=debug_info["projected_short_term"],
                    rgba=color_scheme.EYE_MODEL_OUTLINE_SHORT_TERM_DEBUG.as_float,
                    thickness=2,
                )

            # always draw pupil
            draw_pupil_outline(result, color_rgb=color_scheme.PUPIL_ELLIPSE_3D.as_float)

        if self.__debug_window_button:
            self.__debug_window_button.label = self.__debug_window_button_label

    def cleanup(self):
        # if we change detectors, be sure debug window is also closed
        self.debug_window_close()

    # Public

    def reset_model(self):
        self.detector.reset()

    # Debug window management

    @property
    def __debug_window_button_label(self) -> str:
        if not self.is_debug_window_open:
            return "Open debug window"
        else:
            return "Close debug window"

    @property
    def is_debug_window_open(self) -> bool:
        return self.debugVisualizer3D.window is not None

    def debug_window_toggle(self):
        if not self.is_debug_window_open:
            self.debug_window_open()
        else:
            self.debug_window_close()

    def debug_window_open(self):
        if not self.is_debug_window_open:
            self.debugVisualizer3D.open_window()

    def debug_window_close(self):
        if self.is_debug_window_open:
            self.debugVisualizer3D.close_window()

    def debug_window_update(self):
        if self.is_debug_window_open:
            self.debugVisualizer3D.update_window(
                self.g_pool, self._recent_detection_result
            )
