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
import PIL
from pupil_detector_plugins.utils import get_predictions
from pupil_detector_plugins.models import model_dict
import torchvision
import time

COLOR_MAX = 255
COLOR_CAP = 256
EYE_CLASS = 1
IMAGE_MOD = 16
BBOX_EXTRA_SPACE = 20
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8
EYE_CLASS = 1
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
        """
                기존 __init__에 model_path, device, preview 등을 인자로 추가.
                """
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(properties or {})


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

        self.model = model_dict[model_name]().to(self.device)
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
        init_dict["properties"] = self.detector_2d.get_properties()
        return init_dict

    def detect(self, frame, **kwargs):
        # convert roi-plugin to detector roi
        roi = Roi(*self.g_pool.roi.bounds)

        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None

        # start = time.time()
        result = self.detector_2d.detect(
            gray_img=frame.gray,
            color_img=debug_img,
            roi=roi,
        )


        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )

        # Create basic pupil datum
        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )

        # Fill out 2D model data
        datum["ellipse"] = {}
        datum["ellipse"]["axes"] = result["ellipse"]["axes"]
        datum["ellipse"]["angle"] = result["ellipse"]["angle"]
        datum["ellipse"]["center"] = result["ellipse"]["center"]
        # end = time.time()
        # print(end - start)

        return datum

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

    # def detect_deepVOG(self, frame, **kwargs):
    #     # Assuming frame is preprocessed and contains the deep learning output
    #     # e.g., frame is the output of a deep learning model with shape (height, width, 3)
    #     # Extract mask-confidence from frame
    #
    #     if not isinstance(frame, np.ndarray):  # frame이 NumPy 배열이 아닐 경우
    #         try:
    #             frame = self.convert_mjpeg_to_numpy(frame)
    #         except ValueError as e:
    #             print(f"Error converting MJPEGFrame: {e}")
    #             return None
    #     frame_resized = cv2.resize(frame, (320, 240))
    #     model = deepvog.load_DeepVOG()
    #     # Y_batch = model.predict(frame)
    #     Y_batch = model.predict(np.expand_dims(frame_resized, axis=0))
    #     pred_each = Y_batch[:, :, 1]  # mask-confidence
    #
    #     # Use eyefitter to fit an ellipse and obtain the result
    #     result = self.unproject_single_observation(pred_each)
    #
    #     return result

    # def detect_edgaze(self, frame, **kwargs):
    #     if not isinstance(frame, np.ndarray):
    #         try:
    #             frame = self.convert_mjpeg_to_numpy(frame)
    #         except ValueError as e:
    #             print(f"Error converting MJPEGFrame: {e}")
    #             return None
    #
    #     # Edgaze에서 기본적으로 400(H)×640(W)를 쓰고 싶다면:
    #     # frame_resized = cv2.resize(frame, (640, 400))  # (width=640, height=400)
    #
    #     # 만약 EyeSegmentation이 ndarray 직접 입력을 받는 `predict_image`가 있다면:
    #     Y_batch = self.model.predict_image(frame)
    #
    #     # Y_batch의 shape가 (400, 640, 채널수)처럼 나온다고 가정할 때,
    #     # 예: 채널 1이 confidence map이라면:
    #
    #     # Y_batch_resized = cv2.resize(Y_batch, (192,192))
    #     result = self.unproject_single_observation(Y_batch)
    #     return result

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

    def find_bbox(self, img):
        """find the region most likely to be the eye and find its bbox

        Args:
            img: output from the eye segmentation
        """
        shape = img.shape

        bbox = {"x_min": shape[1], "x_max": 0, "y_min": shape[0], "y_max": 0}

        bboxs = []
        for c in range(shape[1]):
            check = False
            for r in range(shape[0]):
                if img[r, c] >= EYE_CLASS:
                    bbox["x_min"] = min(bbox["x_min"], c)
                    bbox["y_min"] = min(bbox["y_min"], r)
                    bbox["x_max"] = max(bbox["x_max"], c)
                    bbox["y_max"] = max(bbox["y_max"], r)
                    check = True

            if not check and bbox["x_max"] > 0:
                bboxs.append(bbox)
                bbox = {"x_min": shape[1], "x_max": 0, "y_min": shape[0], "y_max": 0}

        if len(bboxs) == 0:
            return {"x_min": 0, "x_max": shape[1], "y_min": 0, "y_max": shape[0]}

        # find the biggest region to be the bbox
        best_bbox = bboxs[0]
        for bbox in bboxs:
            area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])

            best_area = (best_bbox["x_max"] - best_bbox["x_min"]) * (
                best_bbox["y_max"] - best_bbox["y_min"]
            )

            if area > best_area:
                best_bbox = dict(bbox)

        return dict(best_bbox)

    def extract_pupil(self, predict):
        """
            this function extract pupil from segmentation map,
            pupil result is used in the later gaze prediction process.
        """
        predict = np.array(predict)
        bbox = self.find_bbox(predict)
        if np.max(predict) > 0:
            predict = predict / np.max(predict)
        blank_img = np.zeros_like(predict)
        blank_img[
            bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]
        ] = predict[bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]]

        predict = blank_img

        low_pass_filter = predict < EYE_CLASS
        predict[low_pass_filter] = 0

        # if self.preview:
        #     cv2.imshow(name, predict)
        #     cv2.waitKey(30)

        predict = np.expand_dims(predict, axis=0)

        return predict




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

    #################### Hongik IULab ###################################
    def detect_RITnet(self, frame, **kwargs):
        """
        RITnet으로 동공을 검출하고, 결과를 Pupil Labs datum 형식으로 반환하는 예시.
        1) RITnet 세그멘테이션
        2) 동공 라벨 -> 이진 마스크
        3) Contour + fitEllipse
        4) Pupil Labs 결과 dict(datum) 생성
        """

        # ---------- 1) 기본 검사 및 그레이 변환 ----------
        if not isinstance(frame, np.ndarray):
            try:
                img = self.convert_mjpeg_to_numpy(frame)
            except ValueError as e:
                print(f"Error converting MJPEGFrame: {e}")
                return None

        start_time = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)

        # ---------- 2) RITnet 전처리 (감마+CLAHE+Normalize) + 추론 ----------
        img_tensor = self.get_img(gray)  # shape=[1,H,W]
        data = img_tensor.unsqueeze(0).to(self.device)  # shape=[1,1,H,W]

        with torch.no_grad():
            # start_time = time.time()
            output = self.model(data)  # 예: shape=[1,4,H,W]
            # end_time = time.time()
            # print(1 / (end_time - start_time))


        predict = get_predictions(output)  # shape=[1,H,W]
        predict_2d = predict[0].cpu().numpy()  # shape=[H,W], 라벨(0..3)

        # ---------- 3) 동공 라벨(3)만 추출 -> 이진 마스크 (0 or 255) ----------
        pupil_mask = np.zeros_like(predict_2d, dtype=np.uint8)
        # 만약 동공 라벨이 1이라면: pupil_mask[predict_2d == 1] = 255
        pupil_mask[predict_2d == 3] = 255

        # ---------- 4) findContours + fitEllipse ----------
        contours, _ = cv2.findContours(
            pupil_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        end_time = time.time()
        # print(1/(end_time - start_time))

        if not contours:
            # 동공 미검출 -> Pupil Labs 표준 dict
            result = {
                "location": (0.0, 0.0),
                "diameter": 0.0,
                "confidence": 0.0,
                "ellipse": {
                    "axes": (0.0, 0.0),
                    "angle": 0.0,
                    "center": (0.0, 0.0),
                },
            }
        else:
            # 가장 큰 컨투어 선택
            best_contour = max(contours, key=cv2.contourArea)
            if len(best_contour) < 5:
                # fitEllipse는 최소 5개 점 필요
                result = {
                    "location": (0.0, 0.0),
                    "diameter": 0.0,
                    "confidence": 0.0,
                    "ellipse": {
                        "axes": (0.0, 0.0),
                        "angle": 0.0,
                        "center": (0.0, 0.0),
                    },
                }
            else:
                ellipse = cv2.fitEllipse(best_contour)  # ((cx, cy),(MA, ma), angleDeg)
                (cx, cy), (MA, ma), angle_deg = ellipse

                # 간단히 confidence=1.0, 필요시 support ratio 계산 가능
                conf_val = 1.0

                result = {
                    "location": (float(cx), float(cy)),
                    "diameter": float(MA),  # 장축을 diameter로
                    "confidence": conf_val,
                    "ellipse": {
                        "axes": (float(MA), float(ma)),
                        "angle": float(angle_deg),
                        "center": (float(cx), float(cy)),
                    },
                }

        # ---------- 5) Pupil Labs 최종 datum 생성 ----------
        # location -> (px,py), normalize => (x',y') in [0..1]
        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )

        # create_pupil_datum(...) => Pupil Labs 'datum' 형식
        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )

        # ellipse 정보 채워넣기
        datum["ellipse"] = {}
        datum["ellipse"]["axes"] = result["ellipse"]["axes"]
        datum["ellipse"]["angle"] = result["ellipse"]["angle"]
        datum["ellipse"]["center"] = result["ellipse"]["center"]

        return datum
    #################################################################

    def unproject_single_observation(self, prediction, mask=None, threshold=0.5):
        # try:
        #     assert len(prediction.shape) == 2
        #     assert prediction.shape == self.image_shape
        # except(AssertionError):
        #     raise AssertionError(
        #         "Shape of the observation input has to be (image_height, image_width) specified in the initialization of object, or if default, (240,320)")

        # Fit an ellipse from the prediction map
        ellipse_info = fit_ellipse(prediction, mask=mask)
        ellipse_confidence = 0

        if ellipse_info is not None:
            rr, cc, centre, w, h, radian, ell = ellipse_info
            ellipse_confidence = computeEllipseConfidence(prediction, centre, w, h, radian)

            result = {
                'ellipse': {
                    'center': (float(centre[0]), float(centre[1])),
                    'axes': (float(w), float(h)),
                    'angle': float(np.degrees(radian)),  # 라디안을 각도로 변환
                },
                'diameter': float(h),
                'location': (float(centre[0]), float(centre[1])),
                'confidence': float(ellipse_confidence),
            }
        else:
            result = {
                'ellipse': {
                    'center': (0.0, 0.0),
                    'axes': (0.0, 0.0),
                    'angle': 0.0,
                },
                'diameter': 0.0,
                'location': (0.0, 0.0),
                'confidence': 0.0,
            }

        return result



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