#!/usr/bin/env python3
"""Extract offline MapAnything image features for BEVDet.

This script runs MapAnything encoder + info sharing + adapter and stores
per-sample fp16 tensors to disk, so BEVDet can be trained without importing
MapAnything/UniCeption in the BEVDet environment.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm


class MapAnythingAdapter(nn.Module):
    """Same structure as MapAnythingAdapterNeck without mmcv dependency."""

    def __init__(
        self,
        in_channels: int = 1536,
        out_channels: int = 256,
        target_h: int = 16,
        target_w: int = 44,
        num_convs: int = 2,
    ):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_convs - 1):
            layers.extend(
                [
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = F.interpolate(
                x,
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
            )
        return self.proj(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MapAnything features")
    parser.add_argument(
        "--bevdet-root",
        default="/home/batchcom/lr/BEVDet",
        help="BEVDet repo root (used to resolve relative image paths)",
    )
    parser.add_argument(
        "--mapanything-root",
        default="/home/batchcom/lr/map-anything",
        help="map-anything repo root",
    )
    parser.add_argument(
        "--mapanything-ckpt",
        default="/home/batchcom/lr/map-anything/ckpt",
        help="MapAnything checkpoint directory with config.json/model.safetensors",
    )
    parser.add_argument(
        "--train-ann",
        default="/home/batchcom/lr/BEVDet/data/nuscenes/bevdetv3-nuscenes_infos_train.pkl",
    )
    parser.add_argument(
        "--val-ann",
        default="/home/batchcom/lr/BEVDet/data/nuscenes/bevdetv3-nuscenes_infos_val.pkl",
    )
    parser.add_argument(
        "--output-root",
        default="/home/dataset-local/lr/data/mapdepth",
        help="Output directory. Will create train/ and val/ subfolders.",
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        default=[
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ],
    )
    parser.add_argument("--input-h", type=int, default=256)
    parser.add_argument("--input-w", type=int, default=704)
    parser.add_argument("--crop-h-min", type=float, default=0.0)
    parser.add_argument("--crop-h-max", type=float, default=0.0)
    parser.add_argument("--resize-test", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--adapter-seed",
        type=int,
        default=0,
        help="Seed for deterministic adapter initialization",
    )
    return parser.parse_args()


def mmlab_normalize(img: Image.Image) -> torch.Tensor:
    """Match PrepareImageInputs.mmlabNormalize behavior."""
    arr = np.array(img).astype(np.float32)
    # mmcv.imnormalize(..., to_rgb=True) flips channel order.
    arr = arr[..., ::-1]
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


def sample_aug_test(
    src_h: int,
    src_w: int,
    input_h: int,
    input_w: int,
    crop_h_min: float,
    crop_h_max: float,
    resize_test: float,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    resize = float(input_w) / float(src_w)
    resize += resize_test
    resize_dims = (int(src_w * resize), int(src_h * resize))
    new_w, new_h = resize_dims
    crop_h = int((1 - np.mean((crop_h_min, crop_h_max))) * new_h) - input_h
    crop_w = int(max(0, new_w - input_w) / 2)
    crop = (crop_w, crop_h, crop_w + input_w, crop_h + input_h)
    return resize_dims, crop


def preprocess_image(
    img_path: str,
    input_h: int,
    input_w: int,
    crop_h_min: float,
    crop_h_max: float,
    resize_test: float,
) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    resize_dims, crop = sample_aug_test(
        src_h=img.height,
        src_w=img.width,
        input_h=input_h,
        input_w=input_w,
        crop_h_min=crop_h_min,
        crop_h_max=crop_h_max,
        resize_test=resize_test,
    )
    img = img.resize(resize_dims)
    img = img.crop(crop)
    return mmlab_normalize(img)


def resolve_img_path(path: str, bevdet_root: str) -> str:
    if os.path.isabs(path):
        return path
    if path.startswith("./"):
        return os.path.join(bevdet_root, path[2:])
    return os.path.join(bevdet_root, path)


def load_infos(ann_file: str) -> List[Dict]:
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data["infos"]
    return data


def build_mapanything(mapanything_root: str, ckpt_dir: str, device: torch.device):
    if mapanything_root not in sys.path:
        sys.path.insert(0, mapanything_root)

    from mapanything.models.mapanything.model import MapAnything

    config_path = os.path.join(ckpt_dir, "config.json")
    weights_path = os.path.join(ckpt_dir, "model.safetensors")
    with open(config_path, "r") as f:
        config = json.load(f)

    model = MapAnything(**config)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model.encoder, model.info_sharing, model.scale_token


@torch.no_grad()
def forward_mapanything(
    imgs: torch.Tensor,
    encoder: nn.Module,
    info_sharing: nn.Module,
    scale_token: torch.Tensor,
    adapter: nn.Module,
) -> torch.Tensor:
    from uniception.models.encoders.base import ViTEncoderInput
    from uniception.models.info_sharing.base import MultiViewTransformerInput

    bsz, num_views, c, im_h, im_w = imgs.shape
    patch_size = 14

    pad_h = (patch_size - im_h % patch_size) % patch_size
    pad_w = (patch_size - im_w % patch_size) % patch_size
    flat = imgs.permute(1, 0, 2, 3, 4).contiguous().view(num_views * bsz, c, im_h, im_w)
    if pad_h > 0 or pad_w > 0:
        flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="reflect")

    enc_input = ViTEncoderInput(image=flat, data_norm_type="dinov2")
    enc_out = encoder(enc_input)
    feats = list(enc_out.features.chunk(num_views, dim=0))
    regs = None
    if enc_out.registers is not None:
        regs = list(enc_out.registers.chunk(num_views, dim=0))

    scale = scale_token.unsqueeze(0).unsqueeze(-1).repeat(bsz, 1, 1)
    info_input = MultiViewTransformerInput(
        features=feats,
        additional_input_tokens=scale,
        additional_input_tokens_per_view=regs,
    )
    result = info_sharing(info_input)
    final_output = result[0] if isinstance(result, tuple) else result

    x = torch.stack(final_output.features, dim=1)  # (B, N, 1536, H, W)
    x = x.contiguous().view(bsz * num_views, x.shape[2], x.shape[3], x.shape[4])
    x = adapter(x)
    x = x.view(bsz, num_views, x.shape[1], x.shape[2], x.shape[3])
    return x


def save_tensor_atomic(path: Path, tensor: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def process_split(
    split: str,
    ann_file: str,
    out_dir: Path,
    args: argparse.Namespace,
    encoder: nn.Module,
    info_sharing: nn.Module,
    scale_token: torch.Tensor,
    adapter: nn.Module,
    device: torch.device,
):
    infos = load_infos(ann_file)
    if args.max_samples > 0:
        infos = infos[: args.max_samples]

    pbar = tqdm(range(0, len(infos), args.batch_size), desc=f"{split}")
    saved = 0
    skipped = 0
    failed = 0

    for start in pbar:
        batch_infos = infos[start : start + args.batch_size]
        tokens: List[str] = []
        batch_imgs: List[torch.Tensor] = []
        out_paths: List[Path] = []

        for info in batch_infos:
            token = info["token"]
            out_path = out_dir / f"{token}.pt"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            try:
                cams = info["cams"]
                img_tensors = []
                for cam_name in args.cams:
                    cam = cams[cam_name]
                    img_path = resolve_img_path(cam["data_path"], args.bevdet_root)
                    img_tensors.append(
                        preprocess_image(
                            img_path=img_path,
                            input_h=args.input_h,
                            input_w=args.input_w,
                            crop_h_min=args.crop_h_min,
                            crop_h_max=args.crop_h_max,
                            resize_test=args.resize_test,
                        )
                    )
                img = torch.stack(img_tensors, dim=0)  # (N, 3, H, W)
                tokens.append(token)
                out_paths.append(out_path)
                batch_imgs.append(img)
            except Exception as exc:
                failed += 1
                print(f"[WARN] failed preprocessing token={token}: {exc}")

        if not batch_imgs:
            pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)
            continue

        inp = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)  # (B, N, 3, H, W)
        feats = forward_mapanything(
            imgs=inp,
            encoder=encoder,
            info_sharing=info_sharing,
            scale_token=scale_token,
            adapter=adapter,
        )
        feats = feats.half().cpu() if args.fp16 else feats.float().cpu()

        for i, out_path in enumerate(out_paths):
            save_tensor_atomic(out_path, feats[i].contiguous())  # (N, C, H, W)
            saved += 1

        pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)

    print(
        f"[{split}] done. saved={saved}, skipped={skipped}, failed={failed}, total={len(infos)}"
    )


def main():
    args = parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder, info_sharing, scale_token = build_mapanything(
        mapanything_root=args.mapanything_root,
        ckpt_dir=args.mapanything_ckpt,
        device=device,
    )
    encoder.eval()
    info_sharing.eval()

    torch.manual_seed(args.adapter_seed)
    adapter = MapAnythingAdapter(
        in_channels=1536,
        out_channels=256,
        target_h=args.input_h // 16,
        target_w=args.input_w // 16,
        num_convs=2,
    ).to(device)
    adapter.eval()
    print(
        "[WARN] adapter weights are randomly initialized with a fixed seed. "
        "For best quality, replace with a trained adapter before extraction."
    )

    meta = {
        "dtype": "fp16" if args.fp16 else "fp32",
        "shape_per_sample": [len(args.cams), 256, args.input_h // 16, args.input_w // 16],
        "cams": args.cams,
        "input_size": [args.input_h, args.input_w],
        "crop_h": [args.crop_h_min, args.crop_h_max],
        "resize_test": args.resize_test,
        "adapter_seed": args.adapter_seed,
        "train_ann": args.train_ann,
        "val_ann": args.val_ann,
    }
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    process_split(
        split="train",
        ann_file=args.train_ann,
        out_dir=out_root / "train",
        args=args,
        encoder=encoder,
        info_sharing=info_sharing,
        scale_token=scale_token,
        adapter=adapter,
        device=device,
    )
    process_split(
        split="val",
        ann_file=args.val_ann,
        out_dir=out_root / "val",
        args=args,
        encoder=encoder,
        info_sharing=info_sharing,
        scale_token=scale_token,
        adapter=adapter,
        device=device,
    )


if __name__ == "__main__":
    main()

