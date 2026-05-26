import argparse
import json
from pathlib import Path

import torch

from src.config import CHECKPOINTS_DIR, METRICS_DIR, VIDEOS_DIR
from src.inference import FaceMeshDetector, run_video_inference
from src.models import CNNLSTM, SmallEyeCNN
from src.training import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True, help="входное видео (mp4/avi/...)")
    p.add_argument("--model", choices=["ear", "cnn", "lstm"], required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="annotated mp4 (по умолчанию outputs/videos/<stem>_<model>.mp4)")
    p.add_argument("--cnn-ckpt", type=Path, default=CHECKPOINTS_DIR / "small_eye_cnn.pt")
    p.add_argument("--lstm-ckpt", type=Path, default=CHECKPOINTS_DIR / "cnn_lstm.pt")
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    if args.output is None:
        args.output = VIDEOS_DIR / f"{args.video.stem}_{args.model}.mp4"

    cnn_model = None
    lstm_model = None
    if args.model in ("cnn", "lstm"):
        cnn_model = SmallEyeCNN()
        if args.cnn_ckpt.exists():
            cnn_model.load_state_dict(torch.load(args.cnn_ckpt, map_location=device))
            print(f"Loaded CNN checkpoint: {args.cnn_ckpt}")
        else:
            print(f"WARNING: CNN checkpoint not found at {args.cnn_ckpt} (random init)")
    if args.model == "lstm":
        lstm_model = CNNLSTM(cnn_model)
        if args.lstm_ckpt.exists():
            lstm_model.load_state_dict(torch.load(args.lstm_ckpt, map_location=device))
            print(f"Loaded LSTM checkpoint: {args.lstm_ckpt}")
        else:
            print(f"WARNING: LSTM checkpoint not found at {args.lstm_ckpt} (random init)")

    detector = FaceMeshDetector()
    result = run_video_inference(
        video_path=args.video,
        method=args.model,
        output_path=args.output,
        cnn_model=cnn_model,
        lstm_model=lstm_model,
        device=device,
        detector=detector,
    )
    print(json.dumps(result, indent=2))
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = METRICS_DIR / f"{args.video.stem}_{args.model}_inference.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nAnnotated video: {args.output}")
    print(f"Metrics JSON:   {out_json}")


if __name__ == "__main__":
    main()
