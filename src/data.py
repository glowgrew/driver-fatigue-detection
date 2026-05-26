"""Загрузчик FL3D. Бинарная метка: alert -> 0, microsleep|yawning -> 1.
Идентификатор сессии совпадает с субъектом (одна папка = одно лицо)."""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import EYE_SIZE, NTHU_DIR, SEED, SEQ_LEN, SEQ_STEP
from .inference import FaceMeshDetector, crop_eye_region

LABEL_ALERT = 0
LABEL_DROWSY = 1

FL3D_KAGGLE_HANDLE = "matjazmuc/frame-level-driver-drowsiness-detection-fl3d"
DROWSY_STATES = {"microsleep", "yawning"}

_FRAME_IDX = re.compile(r"^frame(\d+)\.jpg$")


@dataclass
class Frame:
    path: Path
    subject: str
    condition: str
    state: str
    frame_idx: int
    label: int


def ensure_fl3d_dataset(local_dir: Path = NTHU_DIR, kaggle_handle: str = FL3D_KAGGLE_HANDLE) -> Path:
    """Локально из data/raw/nthu_ddd/classification_frames/, иначе из кеша kagglehub."""
    local_dir = Path(local_dir)
    if (local_dir / "classification_frames").exists():
        return local_dir
    import kagglehub
    print(f"Downloading FL3D via kagglehub: {kaggle_handle}")
    return Path(kagglehub.dataset_download(kaggle_handle))


def index_fl3d(root: Path | None = None) -> list[Frame]:
    """Сканирует root/classification_frames/, парсит annotations_final.json в каждой сессии.
    При root=None - автозагрузка через kagglehub."""
    if root is None:
        root = ensure_fl3d_dataset()
    root = Path(root)
    frames_root = root / "classification_frames"
    if not frames_root.exists():
        frames_root = root
    frames: list[Frame] = []
    for session_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        ann_path = session_dir / "annotations_final.json"
        if not ann_path.exists():
            continue
        with ann_path.open() as f:
            ann = json.load(f)
        for fname, meta in ann.items():
            state = meta.get("driver_state")
            if state == "alert":
                label = LABEL_ALERT
            elif state in DROWSY_STATES:
                label = LABEL_DROWSY
            else:
                continue
            m = _FRAME_IDX.match(fname)
            if m is None:
                continue
            img_path = session_dir / fname
            if not img_path.exists():
                continue
            frames.append(Frame(
                path=img_path,
                subject=session_dir.name,
                condition="",
                state=state,
                frame_idx=int(m.group(1)),
                label=label,
            ))
    return frames


def subject_wise_split(
    frames: list[Frame],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = SEED,
) -> tuple[list[Frame], list[Frame], list[Frame]]:
    # разбиение по субъектам - один человек не появляется в двух выборках сразу
    subjects = sorted({f.subject for f in frames})
    rng = random.Random(seed); rng.shuffle(subjects)
    n = len(subjects)
    n_train, n_val = max(1, round(n * ratios[0])), max(1, round(n * ratios[1]))
    while n_train + n_val >= n:
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1
    train_s = set(subjects[:n_train])
    val_s = set(subjects[n_train:n_train + n_val])
    train, val, test = [], [], []
    for f in frames:
        if f.subject in train_s:
            train.append(f)
        elif f.subject in val_s:
            val.append(f)
        else:
            test.append(f)
    return train, val, test


def stratified_random_split(frames, ratios=(0.7, 0.15, 0.15), seed=SEED):
    # покадровый split со стратификацией - субъект может протечь между выборками,
    # для честной оценки используем subject_wise_split, это только для сравнения
    groups = {}
    for f in frames:
        groups.setdefault((f.subject, f.state), []).append(f)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for fs in groups.values():
        fs = list(fs); rng.shuffle(fs)
        n = len(fs)
        n_tr = int(n * ratios[0])
        n_vl = int(n * ratios[1])
        train.extend(fs[:n_tr])
        val.extend(fs[n_tr:n_tr + n_vl])
        test.extend(fs[n_tr + n_vl:])
    return train, val, test


def build_clips(frames, seq_len=SEQ_LEN, step=SEQ_STEP):
    # скользящее окно внутри (subject, state), метка - по большинству кадров
    # TODO: короткие сессии (<seq_len) пока пропускаем, padding попробовать
    groups = {}
    for f in frames:
        groups.setdefault((f.subject, f.state), []).append(f)
    clips = []
    for fs in groups.values():
        fs_sorted = sorted(fs, key=lambda x: x.frame_idx)
        if len(fs_sorted) < seq_len:
            continue
        for start in range(0, len(fs_sorted) - seq_len + 1, step):
            window = fs_sorted[start:start + seq_len]
            drowsy = sum(f.label for f in window)
            label = LABEL_DROWSY if drowsy > seq_len // 2 else LABEL_ALERT
            clips.append((window, label))
    return clips


def precompute_eye_crops(frames, cache_path, sides=("left", "right"), overwrite=False):
    """Один прогон MediaPipe по всем кадрам -> .npz {path_str: (2, H, W) uint8}.
    Без кеша эпоха идет час, с кешем - секунды."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    detector = FaceMeshDetector()
    crops = {}
    missed = 0
    for f in tqdm(frames, desc="precompute eye crops"):
        bgr = cv2.imread(str(f.path))
        if bgr is None:
            missed += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lm = detector.detect(rgb)
        if lm is None:
            # mediapipe иногда теряет лицо на сильных поворотах головы
            missed += 1
            continue
        stacked = np.stack([crop_eye_region(bgr, lm, s, EYE_SIZE) for s in sides])
        crops[str(f.path)] = stacked
    detector.close()
    np.savez_compressed(cache_path, **crops)
    print(f"Saved {len(crops)} / {len(frames)} eye crops to {cache_path} (missed {missed})")
    return cache_path


def load_eye_cache(cache_path):
    with np.load(cache_path) as data:
        return {k: data[k] for k in data.files}


def get_eye_transforms(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10),
                     p=0.4, border_mode=cv2.BORDER_REPLICATE),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def _select_crop(stacked, side):
    if side == "left":
        return stacked[0]
    if side == "right":
        return stacked[1]
    return np.maximum(stacked[0], stacked[1])


class NTHUFrameDataset(Dataset):
    def __init__(self, frames, cache, transform=None, side="both"):
        self.frames = [f for f in frames if str(f.path) in cache]
        self.cache = cache
        self.transform = transform or get_eye_transforms(train=False)
        self.side = side

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        f = self.frames[idx]
        crop = _select_crop(self.cache[str(f.path)], self.side)
        return self.transform(image=crop)["image"], f.label


class NTHUClipDataset(Dataset):
    # clip = (T, 1, H, W)
    def __init__(self, clips, cache, transform=None, side="both"):
        self.clips = [
            (window, label) for window, label in clips
            if all(str(f.path) in cache for f in window)
        ]
        self.cache = cache
        self.transform = transform or get_eye_transforms(train=False)
        self.side = side

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        window, label = self.clips[idx]
        tensors = []
        for f in window:
            crop = _select_crop(self.cache[str(f.path)], self.side)
            tensors.append(self.transform(image=crop)["image"])
        return torch.stack(tensors), label


def build_loaders(train_frames, val_frames, test_frames, cache, batch_size=64, clip=False):
    tr_tf = get_eye_transforms(train=True)
    vl_tf = get_eye_transforms(train=False)
    if clip:
        train_ds = NTHUClipDataset(build_clips(train_frames), cache, transform=tr_tf)
        val_ds = NTHUClipDataset(build_clips(val_frames), cache, transform=vl_tf)
        test_ds = NTHUClipDataset(build_clips(test_frames), cache, transform=vl_tf)
    else:
        train_ds = NTHUFrameDataset(train_frames, cache, transform=tr_tf)
        val_ds = NTHUFrameDataset(val_frames, cache, transform=vl_tf)
        test_ds = NTHUFrameDataset(test_frames, cache, transform=vl_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
