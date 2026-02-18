import os
import threading
import time
import numpy as np
import torch
import cv2
from collections import deque
from pupil_detector_plugins import PupilDetectorPlugin
from dv_processing.io import CameraCapture
from dv_processing import EventStreamSlicer
from datetime import timedelta
from pupil_detector_plugins.dvs_models.TDTracker import Model
from pupil_detector_plugins.dvs_metrics import decode_batch_sa_simdr
import tonic.transforms as transforms




class CaerBiasCoarseFine:
    def __init__(self):
        self.coarse_value = 0
        self.fine_value = 0
        self.enabled = True
        self.sex_n = False
        self.type_normal = True
        self.current_level_normal = True


def caer_bias_coarse_fine_generate(cfg: CaerBiasCoarseFine) -> int:
    bv = 0
    if cfg.enabled:
        bv |= 0x01
    if cfg.sex_n:
        bv |= 0x02
    if cfg.type_normal:
        bv |= 0x04
    if cfg.current_level_normal:
        bv |= 0x08
    bv |= (cfg.fine_value & 0xFF) << 4
    bv |= (cfg.coarse_value & 0x07) << 12
    return bv


def caer_bias_coarse_fine_parse(bv: int) -> CaerBiasCoarseFine:
    cfg = CaerBiasCoarseFine()
    cfg.enabled = bool(bv & 0x01)
    cfg.sex_n = bool(bv & 0x02)
    cfg.type_normal = bool(bv & 0x04)
    cfg.current_level_normal = bool(bv & 0x08)
    cfg.fine_value = (bv >> 4) & 0xFF
    cfg.coarse_value = (bv >> 12) & 0x07
    return cfg


class DVSDetectorPlugin(PupilDetectorPlugin):
    def __init__(self, g_pool, **config):
        super().__init__(g_pool)

        # 1) 설정값 로드
        self.TIME_WINDOW_US   = config["TIME_WINDOW_US"]
        self.ORIGINAL_WIDTH   = config["ORIGINAL_WIDTH"]
        self.ORIGINAL_HEIGHT  = config["ORIGINAL_HEIGHT"]
        self.RESIZED_WIDTH    = config["RESIZED_WIDTH"]
        self.RESIZED_HEIGHT   = config["RESIZED_HEIGHT"]
        self.SEQUENCE_LENGTH  = config["SEQUENCE_LENGTH"]
        self.SPATIAL_FACTOR    = config["SPATIAL_FACTOR"]
        self.pixel_tolerances = config["pixel_tolerances"]
        self.GPU_ID           = config["GPU_ID"]
        self.CKPT_PATH        = config["CKPT_PATH"]
        self.WINDOW_NAME      = config.get("WINDOW_NAME", "DVS Preview")

        # 2) 모델 로드
        os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_ID
        args = lambda: None
        args.sensor_width     = self.ORIGINAL_WIDTH
        args.sensor_height    = self.ORIGINAL_HEIGHT
        args.spatial_factor   = self.SPATIAL_FACTOR
        args.pixel_tolerances = self.pixel_tolerances
        self.net = Model(args).cuda().eval()
        state = torch.load(self.CKPT_PATH)
        self.net.load_state_dict(state, strict=False)

        # 3) OpenCV 윈도우 생성
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        # 4) DVS 캡처 & 슬라이서 초기화
        self.capture = CameraCapture()
        self.slicer  = EventStreamSlicer()
        self.slicer.doEveryTimeInterval(
            timedelta(microseconds=self.TIME_WINDOW_US),
            self._slice_callback
        )

        # 5) 내부 버퍼 초기화
        self.buffer_x     = []
        self.buffer_y     = []
        self.buffer_t     = []
        self.buffer_p     = []
        self.start_ts     = None
        self.frame_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.latest_xy   = None
        self.running      = True

        # 6) 이벤트 루프 별도 쓰레드로 실행
        self.thread = threading.Thread(target=self._event_loop, daemon=True)
        self.thread.start()



    def transform_segment(self, xs, ys, ts, ps):
        length = len(xs)
        dtype = [("x","<i8"),("y","<i8"),("t","<i8"),("p","<i8")]
        ev = np.zeros(length, dtype=dtype)
        ev["x"] = (xs * self.RESIZED_WIDTH).astype(np.int64)
        ev["y"] = (ys * self.RESIZED_HEIGHT).astype(np.int64)
        ev["t"] = ts.astype(np.int64)
        ev["p"] = ps.astype(np.int64)
        pipeline = transforms.Compose([
            transforms.ToFrame(sensor_size=(self.RESIZED_WIDTH, self.RESIZED_HEIGHT,2), n_time_bins=4),
            transforms.ToBinaRep(n_frames=1, n_bits=4),
        ])
        frame = pipeline(ev)
        frame_array = frame[0]
        vis = (frame_array.sum(axis=0) / (frame_array.sum(axis=0).max()+1e-6) * 255).astype(np.uint8)
        tensor = torch.from_numpy(frame).unsqueeze(0).float().cuda()
        return tensor, vis, frame_array

    def _slice_callback(self, events):
        ts = events.timestamps().astype(np.int64)
        coords = events.coordinates().astype(np.int64)
        pols = events.polarities().astype(np.int64)

        if self.start_ts is None:
            self.start_ts = ts[0]

        xs = coords[:,0].astype(np.float32) / self.ORIGINAL_WIDTH
        ys = coords[:,1].astype(np.float32) / self.ORIGINAL_HEIGHT

        self.buffer_x.append(xs)
        self.buffer_y.append(ys)
        self.buffer_t.append(ts)
        self.buffer_p.append(pols)

        if ts[-1] - self.start_ts >= self.TIME_WINDOW_US:
            all_x = np.concatenate(self.buffer_x)
            all_y = np.concatenate(self.buffer_y)
            all_t = np.concatenate(self.buffer_t) - self.start_ts
            all_p = np.concatenate(self.buffer_p)

            tensor, vis_frame, frame_array = self.transform_segment(all_x, all_y, all_t, all_p)
            self.frame_buffer.append(frame_array)

            if len(self.frame_buffer) == self.SEQUENCE_LENGTH:
                seq = np.stack(self.frame_buffer, axis=0)
                input_seq = torch.from_numpy(seq).unsqueeze(0).float().cuda()

                with torch.no_grad():
                    pw, ph = self.net(input_seq)

                outputs, _ = decode_batch_sa_simdr(pw, ph)
                x_hat, y_hat = outputs[0, -1]
                self.latest_xy = (float(x_hat), float(y_hat), ts[-1])

                # 시각화
                x_pix = int(x_hat * self.ORIGINAL_WIDTH)
                y_pix = int(y_hat * self.ORIGINAL_HEIGHT)
                vis_resized = cv2.resize(vis_frame,
                                         (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT),
                                         interpolation=cv2.INTER_NEAREST)
                vis_bgr = cv2.cvtColor(vis_resized, cv2.COLOR_GRAY2BGR)
                cv2.circle(vis_bgr, (x_pix, y_pix), 5, (0,255,0), -1)
                cv2.imshow(self.WINDOW_NAME, vis_bgr)
                cv2.waitKey(1)

            # 버퍼 리셋
            self.buffer_x.clear()
            self.buffer_y.clear()
            self.buffer_t.clear()
            self.buffer_p.clear()
            self.start_ts = None

    def _event_loop(self):
        while self.running and self.capture.isRunning():
            ev = self.capture.getNextEventBatch()
            if ev is not None:
                self.slicer.accept(ev)
            time.sleep(0.001)

    def detect(self, frame, timestamp, normalize=True):
        if self.latest_xy is None:
            return None
        x, y, ts = self.latest_xy
        return [(x, y, 1.0)]

    def finish(self):
        self.running = False
        self.thread.join()
        cv2.destroyWindow(self.WINDOW_NAME)
