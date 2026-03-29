from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    experiment_name: str
    protein_csv: str
    function_csv: str
    clinical_csv: str
    slide_feature_dir: str
    checkpoint_dir: str
    protein_id_column: str | None
    function_id_column: str | None
    sample_id_suffix_to_strip: str
    specimen_id_column: str
    slide_id_column: str
    split_group_column: str | None
    region_column: str
    slide_file_suffix: str
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    early_stop_patience: int
    outer_folds: int
    inner_folds: int
    num_workers: int
    seed: int
    device: str
    save_predictions: bool
    verbose: bool
    model_dropout: float
    model_num_attention_heads: int


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        raise ValueError("The YAML configuration root must be a mapping/dictionary.")

    return config_dict


def build_train_config(config_dict: dict[str, Any], cli_verbose: bool = False) -> TrainConfig:
    experiment_cfg = config_dict.get("experiment", {})
    data_cfg = config_dict.get("data", {})
    model_cfg = config_dict.get("model", {})
    training_cfg = config_dict.get("training", {})
    output_cfg = config_dict.get("output", {})

    return TrainConfig(
        experiment_name=experiment_cfg.get("name", "HistoProt"),
        protein_csv=data_cfg.get("protein_csv", ""),
        function_csv=data_cfg.get("function_csv", ""),
        clinical_csv=data_cfg.get("clinical_csv", ""),
        slide_feature_dir=data_cfg.get("slide_feature_dir", ""),
        checkpoint_dir=output_cfg.get("checkpoint_dir", "./checkpoints"),
        protein_id_column=data_cfg.get("protein_id_column"),
        function_id_column=data_cfg.get("function_id_column"),
        sample_id_suffix_to_strip=data_cfg.get("sample_id_suffix_to_strip", ""),
        specimen_id_column=data_cfg.get("specimen_id_column", "specimens_id"),
        slide_id_column=data_cfg.get("slide_id_column", "slides_id"),
        split_group_column=data_cfg.get("split_group_column"),
        region_column=data_cfg.get("region_column", "regions"),
        slide_file_suffix=data_cfg.get("slide_file_suffix", ".feather"),
        epochs=int(training_cfg.get("epochs", 100)),
        batch_size=int(training_cfg.get("batch_size", 4)),
        optimizer=str(training_cfg.get("optimizer", "adamw")).lower(),
        learning_rate=float(training_cfg.get("learning_rate", 5e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-2)),
        warmup_epochs=int(training_cfg.get("warmup_epochs", 10)),
        early_stop_patience=int(training_cfg.get("early_stop_patience", 10)),
        outer_folds=int(training_cfg.get("outer_folds", 5)),
        inner_folds=int(training_cfg.get("inner_folds", 5)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        seed=int(experiment_cfg.get("seed", 0)),
        device=str(experiment_cfg.get("device", "auto")),
        save_predictions=bool(output_cfg.get("save_predictions", False)),
        verbose=bool(experiment_cfg.get("verbose", False) or cli_verbose),
        model_dropout=float(model_cfg.get("dropout", 0.25)),
        model_num_attention_heads=int(model_cfg.get("num_attention_heads", 1)),
    )


def validate_config(config: TrainConfig) -> None:
    required_paths = {
        "protein_csv": config.protein_csv,
        "function_csv": config.function_csv,
        "clinical_csv": config.clinical_csv,
        "slide_feature_dir": config.slide_feature_dir,
    }
    for field_name, field_value in required_paths.items():
        if not field_value:
            raise ValueError(f"`{field_name}` must be provided in config.yaml.")

    for csv_path in (config.protein_csv, config.function_csv, config.clinical_csv):
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Input file does not exist: {csv_path}")

    if not Path(config.slide_feature_dir).exists():
        raise FileNotFoundError(f"Slide feature directory does not exist: {config.slide_feature_dir}")

    if config.batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    if config.epochs <= 0:
        raise ValueError("`epochs` must be positive.")
    if config.learning_rate <= 0:
        raise ValueError("`learning_rate` must be positive.")
    if config.weight_decay < 0:
        raise ValueError("`weight_decay` must be non-negative.")
    if config.outer_folds < 2:
        raise ValueError("`outer_folds` must be at least 2.")
    if config.inner_folds < 2:
        raise ValueError("`inner_folds` must be at least 2.")
    if config.warmup_epochs < 0:
        raise ValueError("`warmup_epochs` must be non-negative.")
    if config.early_stop_patience <= 0:
        raise ValueError("`early_stop_patience` must be positive.")
    if config.optimizer not in {"adamw", "sgd"}:
        raise ValueError("`optimizer` must be either 'adamw' or 'sgd'.")
    if config.model_dropout < 0 or config.model_dropout >= 1:
        raise ValueError("`model.dropout` must be in the interval [0, 1).")
    if config.model_num_attention_heads <= 0:
        raise ValueError("`model.num_attention_heads` must be a positive integer.")


def save_resolved_config(config: TrainConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(asdict(config), file, sort_keys=False, allow_unicode=True)
