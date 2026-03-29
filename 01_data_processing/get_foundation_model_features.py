import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import torch
from huggingface_hub import login
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SUPPORTED_MODELS = ("CONCH", "CONCH_v1_5", "UNI", "UNI_v2", "Virchow2")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class PatchDataset(Dataset):
    def __init__(self, patch_paths: list[Path], preprocess: Callable):
        self.patch_paths = patch_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int):
        patch_path = self.patch_paths[idx]
        with Image.open(patch_path) as image:
            image = image.convert("RGB")
        image = self.preprocess(image)
        return patch_path.name, image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract patch features for each slide under ./dataset/patches using "
            "a selected pathology foundation model."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CONCH",
        choices=SUPPORTED_MODELS,
        help="Foundation model used for patch feature extraction.",
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        default="path/to/patches",
        help="Root directory that contains slide folders of patch images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="path/to/slides_features",
        help=(
            "Root directory to save slide feature feather files. "
            "Outputs are written to <output_dir>/<model>/<slide>.feather."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used for patch feature extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Windows users can keep this at 0.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device, e.g. auto / cuda / cuda:0 / cpu.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face access token. Falls back to env vars or local login cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feather files.",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable automatic mixed precision on CUDA.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token

    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_AUTH_TOKEN"):
        token = os.getenv(env_name)
        if token:
            return token

    return None


def maybe_hf_login(hf_token: str | None) -> None:
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)


def load_conch(hf_token: str | None):
    from conch.open_clip_custom import create_model_from_pretrained

    kwargs = {}
    if hf_token:
        kwargs["hf_auth_token"] = hf_token

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        "hf_hub:MahmoodLab/conch",
        **kwargs,
    )
    return model, preprocess


def load_conch_v1_5(hf_token: str | None):
    from transformers import AutoModel

    kwargs = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token

    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", **kwargs)
    model, preprocess = titan.return_conch()
    return model, preprocess


def load_uni(hf_token: str | None):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess


def load_uni_v2(hf_token: str | None):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess


def load_virchow2(hf_token: str | None):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess


def load_model_and_preprocess(model_name: str, hf_token: str | None):
    loader_map = {
        "CONCH": load_conch,
        "CONCH_v1_5": load_conch_v1_5,
        "UNI": load_uni,
        "UNI_v2": load_uni_v2,
        "Virchow2": load_virchow2,
    }
    return loader_map[model_name](hf_token)


def call_encode_image(model, batch_images: torch.Tensor) -> torch.Tensor:
    try:
        return model.encode_image(batch_images, proj_contrast=False, normalize=False)
    except TypeError:
        return model.encode_image(batch_images)


def extract_conch_features(model, batch_images: torch.Tensor) -> torch.Tensor:
    return call_encode_image(model, batch_images)


def extract_conch_v1_5_features(model, batch_images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):
        return call_encode_image(model, batch_images)

    outputs = model(batch_images)
    if isinstance(outputs, dict):
        for key in ("image_features", "features", "pooler_output", "last_hidden_state"):
            if key in outputs:
                outputs = outputs[key]
                break
        else:
            raise RuntimeError("Unable to find CONCH_v1_5 image features in model outputs.")
    elif isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    if outputs.ndim == 3:
        return outputs[:, 0]

    return outputs


def extract_forward_features(model, batch_images: torch.Tensor) -> torch.Tensor:
    outputs = model(batch_images)
    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs:
            outputs = outputs["last_hidden_state"]
        elif "pooler_output" in outputs:
            outputs = outputs["pooler_output"]
        else:
            outputs = next(iter(outputs.values()))
    elif isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    return outputs


def extract_virchow2_features(model, batch_images: torch.Tensor) -> torch.Tensor:
    outputs = extract_forward_features(model, batch_images)
    if outputs.ndim != 3:
        raise RuntimeError(
            "Virchow2 forward output is expected to be a token sequence with shape [B, N, C]."
        )

    cls_token = outputs[:, 0]
    patch_tokens = outputs[:, 5:]
    return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


def get_feature_extractor(model_name: str):
    extractor_map = {
        "CONCH": extract_conch_features,
        "CONCH_v1_5": extract_conch_v1_5_features,
        "UNI": extract_forward_features,
        "UNI_v2": extract_forward_features,
        "Virchow2": extract_virchow2_features,
    }
    return extractor_map[model_name]


def collect_slide_dirs(patches_dir: Path) -> list[Path]:
    if not patches_dir.exists():
        raise FileNotFoundError(f"Patches directory does not exist: {patches_dir}")

    slide_dirs = [path for path in sorted(patches_dir.iterdir()) if path.is_dir()]
    if not slide_dirs:
        raise RuntimeError(f"No slide directories found in: {patches_dir}")
    return slide_dirs


def collect_patch_paths(slide_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(slide_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def write_slide_features(slide_features: pd.DataFrame, save_path: Path) -> None:
    slide_features.index.name = "patches_name"
    table = pa.Table.from_pandas(slide_features, preserve_index=True)
    feather.write_feather(table, save_path)


def extract_slide_features(
    slide_dir: Path,
    save_path: Path,
    preprocess: Callable,
    model,
    feature_extractor: Callable,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> None:
    patch_paths = collect_patch_paths(slide_dir)
    if not patch_paths:
        print(f"[skip] {slide_dir.name}: no patch images found.")
        return

    dataset = PatchDataset(patch_paths, preprocess)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    patch_names: list[str] = []
    patch_features: list[list[float]] = []

    with torch.inference_mode():
        for batch_names, batch_images in data_loader:
            batch_images = batch_images.to(device, non_blocking=device.type == "cuda")
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda" and use_amp
                else nullcontext()
            )
            with amp_context:
                embeddings = feature_extractor(model, batch_images)
            embeddings = embeddings.detach().float().cpu().numpy()

            patch_names.extend(list(batch_names))
            patch_features.extend(embeddings.tolist())

    slide_features = pd.DataFrame(patch_features, index=patch_names)
    write_slide_features(slide_features, save_path)


def main() -> None:
    args = parse_args()

    hf_token = resolve_hf_token(args.hf_token)
    maybe_hf_login(hf_token)

    patches_dir = Path(args.patches_dir).resolve()
    output_dir = (Path(args.output_dir) / args.model).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    use_amp = not args.disable_amp

    print(f"Model: {args.model}")
    print(f"Patches dir: {patches_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    model, preprocess = load_model_and_preprocess(args.model, hf_token)
    model = model.to(device)
    model.eval()
    feature_extractor = get_feature_extractor(args.model)

    slide_dirs = collect_slide_dirs(patches_dir)
    total_slides = len(slide_dirs)

    for slide_idx, slide_dir in enumerate(slide_dirs, start=1):
        save_path = output_dir / f"{slide_dir.name}.feather"
        if save_path.exists() and not args.overwrite:
            print(f"[skip] {slide_idx}/{total_slides} {slide_dir.name}: {save_path.name} exists.")
            continue

        print(f"[run]  {slide_idx}/{total_slides} {slide_dir.name}")
        extract_slide_features(
            slide_dir=slide_dir,
            save_path=save_path,
            preprocess=preprocess,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
        )
        print(f"[save] {save_path}")


if __name__ == "__main__":
    main()
