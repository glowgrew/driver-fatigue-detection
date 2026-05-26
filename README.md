# Driver Fatigue Detection

Студенческая ML/CV-работа: детекция усталости водителя по видео. Сравнение
rule-based baseline (EAR/PERCLOS) с двумя нейросетевыми подходами
(SmallEyeCNN и CNN+LSTM) на датасете FL3D.

## Архитектура

Pipeline: `видео → MediaPipe Face Mesh → 6-точечные landmarks глаз → один из 3 методов → drowsy alert`.

| # | Метод | Что делает |
|---|-------|------------|
| 1 | EAR + PERCLOS | EAR < 0.21 → закрыт; PERCLOS > 0.35 в окне 90 кадров → drowsy |
| 2 | SmallEyeCNN + PERCLOS | CNN классифицирует eye-crop 24×24 как open/closed; та же PERCLOS-логика |
| 3 | CNN+LSTM (основной) | Frozen SmallEyeCNN извлекает фичи из T=10 кадров → BiLSTM выдаёт drowsy/alert |

```
driver_fatigue/
├── src/
│   ├── config.py       # SEED, paths, hyperparams
│   ├── data.py         # FL3D dataset, subject-wise split, transforms
│   ├── models.py       # SmallEyeCNN, CNNLSTM
│   ├── training.py     # train_loop, EarlyStopping, metrics, plots
│   └── inference.py    # MediaPipe wrapper, EAR, eye-crop, run_video_inference
├── data/
│   ├── raw/fl3d/       # FL3D (auto-download, gitignored)
│   ├── raw/test_videos/
│   ├── cache/          # eye_crops.npz
│   └── models/         # face_landmarker.task
├── outputs/
│   ├── checkpoints/    # *.pt
│   ├── plots/          # training curves, confusion matrices
│   ├── metrics/        # results.json
│   └── videos/         # annotated demo mp4
├── notebooks/driver_fatigue_detection.ipynb
├── main.py             # CLI
└── pyproject.toml
```

## Tech stack

- Python 3.13, uv для зависимостей
- PyTorch 2.12 + torchvision, MPS backend на Apple Silicon
- MediaPipe 0.10.35 (Face Landmarker для 6-точечных landmarks глаз)
- OpenCV для видео I/O, albumentations для аугментаций
- scikit-learn для метрик и confusion matrix
- kagglehub для авто-download FL3D

## Установка

```bash
cd driver_fatigue
uv sync
```

Датасет FL3D (599 МБ, 44 субъекта, 53k размеченных кадров) подтягивается через
`kagglehub` при первом вызове `ensure_fl3d_dataset()` из `src.data`. Источник:
[`matjazmuc/frame-level-driver-drowsiness-detection-fl3d`](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d).

Для доступа к Kaggle нужен `kaggle.json` в `~/.config/kaggle/` (создаётся через
Kaggle → Account → Create New API Token).

Перед первым обучением запускается `precompute_eye_crops(frames, EYE_CACHE_PATH)`
— на выходе `data/cache/eye_crops.npz` ~67 МБ с готовыми (24, 24) crops для обоих
глаз каждого кадра. Без кэша одна эпоха идёт ~1 час, с кэшем — секунды.

## Запуск

Полный пайплайн через notebook:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
    notebooks/driver_fatigue_detection.ipynb \
    --output notebooks/driver_fatigue_detection.ipynb \
    --ExecutePreprocessor.timeout=7200
```

CLI на одном видео:

```bash
# Baseline без обучения (только формула EAR)
uv run python main.py --video data/raw/test_videos/sample.mp4 --model ear

# SmallEyeCNN — требуется checkpoint outputs/checkpoints/small_eye_cnn.pt
uv run python main.py --video data/raw/test_videos/sample.mp4 --model cnn

# CNN+LSTM, основной метод — требуются оба checkpoint'а
uv run python main.py --video data/raw/test_videos/sample.mp4 --model lstm
```

Результат: аннотированное mp4 в `outputs/videos/` + JSON метрики в `outputs/metrics/`.
