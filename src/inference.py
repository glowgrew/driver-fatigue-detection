import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .config import (
    EAR_THRESH,
    EYE_SIZE,
    FACE_MODEL_PATH,
    LEFT_EYE_LANDMARKS,
    MAR_THRESH,
    MOUTH_LANDMARKS,
    PERCLOS_THRESH,
    PERCLOS_WINDOW,
    RIGHT_EYE_LANDMARKS,
    SEQ_LEN,
    YAWN_THRESH,
    YAWN_WINDOW,
)

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_face_landmarker(model_path: Path = FACE_MODEL_PATH) -> Path:
    """Скачивает face_landmarker.task при отсутствии."""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe face landmarker -> {model_path}")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, model_path)
    return model_path


class FaceMeshDetector:
    """Обёртка над mediapipe.tasks.vision.FaceLandmarker (API 0.10.x).
    Индексы ключевых точек те же, что у старого Face Mesh - EAR-формула не меняется."""

    def __init__(self, model_path: Path | None = None, min_det_conf: float = 0.5):
        path = ensure_face_landmarker(model_path or FACE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_det_conf,
            min_face_presence_confidence=min_det_conf,
        )
        self._det = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._det.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def close(self) -> None:
        self._det.close()


def _eye_ar(pts):
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * horizontal))


def compute_ear(landmarks):
    """EAR по Soukupova-Cech 2016: (||p2-p6||+||p3-p5||) / (2*||p1-p4||)."""
    left = landmarks[list(LEFT_EYE_LANDMARKS)]
    right = landmarks[list(RIGHT_EYE_LANDMARKS)]
    return _eye_ar(left), _eye_ar(right)


def compute_mar(landmarks):
    # та же 6-точечная формула, но для рта; >0.5 = широко открыт
    return _eye_ar(landmarks[list(MOUTH_LANDMARKS)])


def crop_eye_region(frame, landmarks, side, out_size=EYE_SIZE, pad_ratio=0.35):
    idx = LEFT_EYE_LANDMARKS if side == "left" else RIGHT_EYE_LANDMARKS
    pts = landmarks[list(idx)]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    side_len = max(x_max - x_min, y_max - y_min) * (1 + pad_ratio)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x0 = max(0, int(cx - side_len / 2))
    y0 = max(0, int(cy - side_len / 2))
    x1 = min(frame.shape[1], int(cx + side_len / 2))
    y1 = min(frame.shape[0], int(cy + side_len / 2))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)


def annotate_frame(frame, bbox, eye_crops, score, score_label, perclos, is_drowsy,
                   mar=None, yawn_ratio=None, is_yawning=False):
    out = frame.copy()
    h, w = out.shape[:2]
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if eye_crops is not None:
        for i, eye in enumerate(eye_crops):
            eye_rgb = cv2.cvtColor(eye, cv2.COLOR_GRAY2BGR)
            eye_big = cv2.resize(eye_rgb, (96, 96), interpolation=cv2.INTER_NEAREST)
            x_off = 10 + i * 110
            out[10:106, x_off:x_off + 96] = eye_big
    cv2.putText(out, f"{score_label}: {score:.3f}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    bar_x, bar_y, bar_w, bar_h = 10, 155, 200, 18
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (200, 200, 200), 1)
    fill = int(bar_w * min(perclos, 1.0))
    color = (0, 0, 255) if is_drowsy else (0, 200, 0)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    cv2.putText(out, f"PERCLOS: {perclos:.2f}", (bar_x + 4, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    if mar is not None:
        cv2.putText(out, f"MAR: {mar:.3f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if yawn_ratio is not None:
        ybar_y = 225
        cv2.rectangle(out, (bar_x, ybar_y), (bar_x + bar_w, ybar_y + bar_h),
                      (200, 200, 200), 1)
        yfill = int(bar_w * min(yawn_ratio, 1.0))
        ycolor = (0, 165, 255) if is_yawning else (0, 200, 0)
        cv2.rectangle(out, (bar_x, ybar_y), (bar_x + yfill, ybar_y + bar_h),
                      ycolor, -1)
        cv2.putText(out, f"YAWN: {yawn_ratio:.2f}", (bar_x + 4, ybar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    if is_drowsy:
        cv2.putText(out, "DROWSY", (w - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    if is_yawning:
        cv2.putText(out, "YAWN", (w - 200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
    return out


class EARBaseline:
    """EAR < порога = глаз закрыт; доля закрытых в окне > порога = сонлив (PERCLOS)."""

    def __init__(self, ear_thresh=EAR_THRESH, perclos_window=PERCLOS_WINDOW, perclos_thresh=PERCLOS_THRESH):
        self.ear_thresh = ear_thresh
        self.perclos_thresh = perclos_thresh
        self._buffer = deque(maxlen=perclos_window)

    def reset(self):
        self._buffer.clear()

    def step(self, ear):
        closed = ear < self.ear_thresh
        self._buffer.append(int(closed))
        perclos = sum(self._buffer) / len(self._buffer)
        return closed, perclos, perclos > self.perclos_thresh


class YawnDetector:
    # MAR > порога = рот открыт, окно усредняет
    def __init__(self, mar_thresh=MAR_THRESH, window=YAWN_WINDOW, thresh=YAWN_THRESH):
        self.mar_thresh = mar_thresh
        self.thresh = thresh
        self._buffer = deque(maxlen=window)

    def reset(self):
        self._buffer.clear()

    def step(self, mar):
        open_mouth = mar > self.mar_thresh
        self._buffer.append(int(open_mouth))
        ratio = sum(self._buffer) / len(self._buffer)
        return open_mouth, ratio, ratio > self.thresh


def run_video_inference(
    video_path: Path,
    method: str,
    output_path: Path | None = None,
    cnn_model: torch.nn.Module | None = None,
    lstm_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    detector: FaceMeshDetector | None = None,
) -> dict:
    if device is None:
        device = torch.device("cpu")
    if detector is None:
        detector = FaceMeshDetector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer: cv2.VideoWriter | None = None
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (w_in, h_in))

    baseline = EARBaseline() if method == "ear" else None
    yawn_detector = YawnDetector()
    perclos_buffer: deque[int] = deque(maxlen=PERCLOS_WINDOW)
    seq_buffer: deque[np.ndarray] = deque(maxlen=SEQ_LEN)

    if method == "cnn":
        if cnn_model is None:
            raise ValueError("cnn_model required for method='cnn'")
        cnn_model.eval().to(device)
    if method == "lstm":
        if lstm_model is None:
            raise ValueError("lstm_model required for method='lstm'")
        lstm_model.eval().to(device)

    n_frames = 0
    n_drowsy = 0
    n_yawning = 0
    n_face_missing = 0
    t_start = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        n_frames += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect(frame_rgb)
        is_drowsy = False
        is_yawning = False
        score = 0.0
        score_label = "--"
        bbox = None
        eye_crops_pair: tuple[np.ndarray, np.ndarray] | None = None
        perclos = 0.0
        mar: float | None = None
        yawn_ratio: float | None = None

        if landmarks is None:
            n_face_missing += 1
        else:
            mar = compute_mar(landmarks)
            _open, yawn_ratio, is_yawning = yawn_detector.step(mar)
            x_min, y_min = landmarks.min(axis=0)
            x_max, y_max = landmarks.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
            left_crop = crop_eye_region(frame_bgr, landmarks, "left")
            right_crop = crop_eye_region(frame_bgr, landmarks, "right")
            eye_crops_pair = (left_crop, right_crop)

            if method == "ear":
                left_ear, right_ear = compute_ear(landmarks)
                mean_ear = (left_ear + right_ear) / 2
                _closed, perclos, is_drowsy = baseline.step(mean_ear)
                score = mean_ear
                score_label = "EAR"
            elif method == "cnn":
                eyes = np.stack([left_crop, right_crop]).astype(np.float32) / 255.0
                eyes = (eyes - 0.5) / 0.5  # та же нормализация, что и при обучении (mean=0.5, std=0.5)
                eyes_t = torch.from_numpy(eyes).unsqueeze(1).to(device)
                with torch.no_grad():
                    logits = cnn_model(eyes_t)
                    p_closed = torch.softmax(logits, dim=1)[:, 1].mean().item()
                closed = p_closed > 0.5
                perclos_buffer.append(int(closed))
                perclos = sum(perclos_buffer) / len(perclos_buffer)
                is_drowsy = perclos > PERCLOS_THRESH
                score = p_closed
                score_label = "P(closed)"
            elif method == "lstm":
                # merged = left_crop  # was using left only first, but max() works better on profile faces
                merged = np.maximum(left_crop, right_crop).astype(np.float32) / 255.0
                merged = (merged - 0.5) / 0.5  # та же нормализация что и при обучении
                seq_buffer.append(merged)
                if len(seq_buffer) == SEQ_LEN:
                    seq = np.stack(list(seq_buffer))[None, :, None, :, :]
                    seq_t = torch.from_numpy(seq).to(device)
                    with torch.no_grad():
                        logits = lstm_model(seq_t)
                        probs = torch.softmax(logits, dim=1)[0]
                    p_drowsy = probs[1].item()
                    is_drowsy = p_drowsy > 0.5
                    score = p_drowsy
                    score_label = "P(drowsy)"

        if is_drowsy:
            n_drowsy += 1
        if is_yawning:
            n_yawning += 1
        if writer is not None:
            out_frame = annotate_frame(frame_bgr, bbox, eye_crops_pair, score,
                                        score_label, perclos, is_drowsy,
                                        mar, yawn_ratio, is_yawning)
            writer.write(out_frame)

    elapsed = time.time() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    detector.close()

    return {
        "method": method,
        "n_frames": n_frames,
        "n_drowsy_frames": n_drowsy,
        "drowsy_ratio": n_drowsy / n_frames if n_frames else 0.0,
        "n_yawning_frames": n_yawning,
        "yawning_ratio": n_yawning / n_frames if n_frames else 0.0,
        "n_face_missing": n_face_missing,
        "fps": n_frames / elapsed if elapsed > 0 else 0.0,
        "elapsed_sec": elapsed,
    }
