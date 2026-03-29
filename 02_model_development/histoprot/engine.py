from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from .model import HistoProt

from .config import TrainConfig
from .data import (
    build_dataloader,
    build_slide_dataset,
    prepare_training_inputs,
    select_specimens_for_groups,
    subset_workspace,
)
from .metrics import compute_mean_feature_correlation
from .runtime import resolve_device, set_random_seed


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    mean_correlation: float
    mean_loss: float
    predictions: pd.DataFrame


def build_model(
    input_dim: int,
    num_proteins: int,
    num_functions: int,
    num_attention_heads: int,
    device: torch.device,
    dropout: float,
) -> HistoProt:
    model = HistoProt(
        input_dim=input_dim,
        num_proteins=num_proteins,
        num_functions=num_functions,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
    )
    return model.to(device)


def build_optimizer(model: HistoProt, optimizer_name: str, learning_rate: float, weight_decay: float) -> Any:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: Any, epochs: int, warmup_epochs: int) -> Any:
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    if warmup_epochs <= 0:
        return cosine_scheduler
    return GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=warmup_epochs,
        after_scheduler=cosine_scheduler,
    )


def compute_batch_loss(
    protein_targets: list[torch.Tensor],
    function_targets: list[torch.Tensor],
    protein_predictions: list[torch.Tensor],
    function_predictions: list[torch.Tensor],
    hierarchical_losses: list[torch.Tensor],
    mse_loss: nn.Module,
) -> torch.Tensor:
    protein_target_tensor = torch.cat(protein_targets, dim=0)
    function_target_tensor = torch.cat(function_targets, dim=0)
    protein_prediction_tensor = torch.cat(protein_predictions, dim=0)
    function_prediction_tensor = torch.cat(function_predictions, dim=0)
    hierarchical_loss = torch.stack(hierarchical_losses).sum()

    return (
        mse_loss(protein_prediction_tensor, protein_target_tensor)
        + mse_loss(function_prediction_tensor, function_target_tensor)
        + hierarchical_loss
    )


def unpack_model_outputs(model_outputs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(model_outputs, dict):
        required_keys = {
            "slide_protein_predictions",
            "slide_function_predictions",
            "hierarchical_consistency_loss",
        }
        missing_keys = required_keys.difference(model_outputs)
        if missing_keys:
            missing_text = ", ".join(sorted(missing_keys))
            raise KeyError(f"Model output dictionary is missing required keys: {missing_text}")
        return (
            model_outputs["slide_protein_predictions"],
            model_outputs["slide_function_predictions"],
            model_outputs["hierarchical_consistency_loss"],
        )

    if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 3:
        return model_outputs[0], model_outputs[1], model_outputs[2]

    raise TypeError(
        "Model forward output must be either a dict with prediction/loss tensors "
        "or a tuple/list of three tensors."
    )


def train_one_epoch(
    model: HistoProt,
    data_loader: DataLoader,
    optimizer: Any,
    accumulation_steps: int,
    device: torch.device,
) -> float:
    model.train()
    mse_loss = nn.MSELoss()
    epoch_losses = []

    accumulated_protein_targets = []
    accumulated_function_targets = []
    accumulated_protein_predictions = []
    accumulated_function_predictions = []
    accumulated_hierarchical_losses = []

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(data_loader, start=1):
        patch_features = batch["patch_features"][0].to(device)
        patch_regions = batch["patch_regions"][0].to(device)
        protein_targets = batch["protein_targets"].to(device)
        function_targets = batch["function_targets"].to(device)

        model_outputs = model(
            patch_features,
            patch_regions,
        )
        protein_predictions, function_predictions, hierarchical_loss = unpack_model_outputs(model_outputs)

        accumulated_protein_targets.append(protein_targets)
        accumulated_function_targets.append(function_targets)
        accumulated_protein_predictions.append(protein_predictions)
        accumulated_function_predictions.append(function_predictions)
        accumulated_hierarchical_losses.append(hierarchical_loss)

        should_update = step % accumulation_steps == 0 or step == len(data_loader)
        if not should_update:
            continue

        loss = compute_batch_loss(
            protein_targets=accumulated_protein_targets,
            function_targets=accumulated_function_targets,
            protein_predictions=accumulated_protein_predictions,
            function_predictions=accumulated_function_predictions,
            hierarchical_losses=accumulated_hierarchical_losses,
            mse_loss=mse_loss,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        epoch_losses.append(loss.item())
        accumulated_protein_targets.clear()
        accumulated_function_targets.clear()
        accumulated_protein_predictions.clear()
        accumulated_function_predictions.clear()
        accumulated_hierarchical_losses.clear()

    return float(np.mean(epoch_losses)) if epoch_losses else float("nan")


def evaluate_model(
    model: HistoProt,
    target_workspace: pd.DataFrame,
    data_loader: DataLoader,
    accumulation_steps: int,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    mse_loss = nn.MSELoss()
    evaluation_losses = []
    prediction_columns: dict[str, np.ndarray] = {}

    accumulated_protein_targets = []
    accumulated_function_targets = []
    accumulated_protein_predictions = []
    accumulated_function_predictions = []
    accumulated_hierarchical_losses = []

    with torch.no_grad():
        for step, batch in enumerate(data_loader, start=1):
            specimen_id = batch["specimen_ids"][0]
            patch_features = batch["patch_features"][0].to(device)
            patch_regions = batch["patch_regions"][0].to(device)
            protein_targets = batch["protein_targets"].to(device)
            function_targets = batch["function_targets"].to(device)

            model_outputs = model(
                patch_features,
                patch_regions,
            )
            protein_predictions, function_predictions, hierarchical_loss = unpack_model_outputs(model_outputs)

            accumulated_protein_targets.append(protein_targets)
            accumulated_function_targets.append(function_targets)
            accumulated_protein_predictions.append(protein_predictions)
            accumulated_function_predictions.append(function_predictions)
            accumulated_hierarchical_losses.append(hierarchical_loss)
            prediction_columns[specimen_id] = protein_predictions.squeeze(0).cpu().numpy()

            should_finalize = step % accumulation_steps == 0 or step == len(data_loader)
            if not should_finalize:
                continue

            loss = compute_batch_loss(
                protein_targets=accumulated_protein_targets,
                function_targets=accumulated_function_targets,
                protein_predictions=accumulated_protein_predictions,
                function_predictions=accumulated_function_predictions,
                hierarchical_losses=accumulated_hierarchical_losses,
                mse_loss=mse_loss,
            )
            evaluation_losses.append(loss.item())
            accumulated_protein_targets.clear()
            accumulated_function_targets.clear()
            accumulated_protein_predictions.clear()
            accumulated_function_predictions.clear()
            accumulated_hierarchical_losses.clear()

    prediction_workspace = pd.DataFrame(prediction_columns, index=target_workspace.index)
    prediction_workspace = prediction_workspace.loc[:, target_workspace.columns]
    mean_correlation = compute_mean_feature_correlation(target_workspace, prediction_workspace)
    mean_loss = float(np.mean(evaluation_losses)) if evaluation_losses else float("nan")
    return EvaluationResult(
        mean_correlation=mean_correlation,
        mean_loss=mean_loss,
        predictions=prediction_workspace,
    )


def save_prediction_tables(
    output_dir: Path,
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    train_predictions.to_csv(output_dir / "train_predictions.csv")
    validation_predictions.to_csv(output_dir / "validation_predictions.csv")
    test_predictions.to_csv(output_dir / "test_predictions.csv")


def save_best_checkpoint(
    checkpoint_path: Path,
    model: HistoProt,
    optimizer: Any,
    epoch: int,
    validation_result: EvaluationResult,
) -> None:
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_corr": validation_result.mean_correlation,
            "validation_loss": validation_result.mean_loss,
        },
        checkpoint_path,
    )


def train_nested_cross_validation(config: TrainConfig, run_dir: Path) -> None:
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    prepared_inputs = prepare_training_inputs(config)

    if len(prepared_inputs.group_labels) < max(config.outer_folds, config.inner_folds):
        raise ValueError("Not enough groups are available for the requested nested cross-validation splits.")

    summary_records = []
    outer_splitter = KFold(n_splits=config.outer_folds, shuffle=True, random_state=config.seed)

    for outer_fold, (development_index, test_index) in enumerate(outer_splitter.split(prepared_inputs.group_labels)):
        development_groups = np.array(prepared_inputs.group_labels)[development_index].tolist()
        test_groups = np.array(prepared_inputs.group_labels)[test_index].tolist()

        test_specimens = select_specimens_for_groups(
            clinical_table=prepared_inputs.clinical_table,
            selected_groups=test_groups,
            split_group_column=config.split_group_column,
            specimen_id_column=config.specimen_id_column,
        )
        protein_test_workspace = subset_workspace(prepared_inputs.protein_targets, test_specimens)
        function_test_workspace = subset_workspace(prepared_inputs.function_targets, test_specimens)

        inner_splitter = KFold(
            n_splits=config.inner_folds,
            shuffle=True,
            random_state=config.seed + outer_fold,
        )
        for inner_fold, (train_index, validation_index) in enumerate(inner_splitter.split(development_groups)):
            set_random_seed(config.seed + outer_fold + inner_fold)

            train_groups = np.array(development_groups)[train_index].tolist()
            validation_groups = np.array(development_groups)[validation_index].tolist()

            train_specimens = select_specimens_for_groups(
                clinical_table=prepared_inputs.clinical_table,
                selected_groups=train_groups,
                split_group_column=config.split_group_column,
                specimen_id_column=config.specimen_id_column,
            )
            validation_specimens = select_specimens_for_groups(
                clinical_table=prepared_inputs.clinical_table,
                selected_groups=validation_groups,
                split_group_column=config.split_group_column,
                specimen_id_column=config.specimen_id_column,
            )

            protein_train_workspace = subset_workspace(prepared_inputs.protein_targets, train_specimens)
            function_train_workspace = subset_workspace(prepared_inputs.function_targets, train_specimens)
            protein_validation_workspace = subset_workspace(prepared_inputs.protein_targets, validation_specimens)
            function_validation_workspace = subset_workspace(prepared_inputs.function_targets, validation_specimens)

            train_dataset = build_slide_dataset(
                protein_targets=protein_train_workspace,
                function_targets=function_train_workspace,
                clinical_table=prepared_inputs.clinical_table,
                config=config,
            )
            validation_dataset = build_slide_dataset(
                protein_targets=protein_validation_workspace,
                function_targets=function_validation_workspace,
                clinical_table=prepared_inputs.clinical_table,
                config=config,
            )
            test_dataset = build_slide_dataset(
                protein_targets=protein_test_workspace,
                function_targets=function_test_workspace,
                clinical_table=prepared_inputs.clinical_table,
                config=config,
            )

            train_loader = build_dataloader(train_dataset, shuffle=True, num_workers=config.num_workers)
            validation_loader = build_dataloader(validation_dataset, shuffle=False, num_workers=config.num_workers)
            test_loader = build_dataloader(test_dataset, shuffle=False, num_workers=config.num_workers)

            input_dim = train_dataset[0]["patch_features"].shape[1]
            model = build_model(
                input_dim=input_dim,
                num_proteins=prepared_inputs.protein_targets.shape[0],
                num_functions=prepared_inputs.function_targets.shape[0],
                num_attention_heads=config.model_num_attention_heads,
                device=device,
                dropout=config.model_dropout,
            )
            optimizer = build_optimizer(
                model=model,
                optimizer_name=config.optimizer,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            scheduler = build_scheduler(
                optimizer=optimizer,
                epochs=config.epochs,
                warmup_epochs=config.warmup_epochs,
            )

            fold_output_dir = run_dir / f"outer_{outer_fold}" / f"inner_{inner_fold}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = fold_output_dir / "checkpoint_best_validation.pth"

            LOGGER.info(
                "Starting outer fold %d / inner fold %d with %d train, %d validation, %d test specimens.",
                outer_fold,
                inner_fold,
                len(train_dataset),
                len(validation_dataset),
                len(test_dataset),
            )

            best_validation_corr = np.nan
            best_epoch = -1
            early_stop_counter = 0

            for epoch in range(config.epochs):
                training_loss = train_one_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    accumulation_steps=config.batch_size,
                    device=device,
                )
                validation_result = evaluate_model(
                    model=model,
                    target_workspace=protein_validation_workspace,
                    data_loader=validation_loader,
                    accumulation_steps=config.batch_size,
                    device=device,
                )
                test_result = evaluate_model(
                    model=model,
                    target_workspace=protein_test_workspace,
                    data_loader=test_loader,
                    accumulation_steps=config.batch_size,
                    device=device,
                )

                LOGGER.info(
                    "Outer %d | Inner %d | Epoch %d | train_loss=%.6f | val_corr=%.6f | val_loss=%.6f | test_corr=%.6f | test_loss=%.6f",
                    outer_fold,
                    inner_fold,
                    epoch,
                    training_loss,
                    validation_result.mean_correlation,
                    validation_result.mean_loss,
                    test_result.mean_correlation,
                    test_result.mean_loss,
                )

                should_save_checkpoint = best_epoch == -1 or (
                    not np.isnan(validation_result.mean_correlation)
                    and (
                        np.isnan(best_validation_corr)
                        or validation_result.mean_correlation > best_validation_corr
                    )
                )

                if should_save_checkpoint:
                    best_validation_corr = validation_result.mean_correlation
                    best_epoch = epoch
                    early_stop_counter = 0
                    save_best_checkpoint(
                        checkpoint_path=best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        validation_result=validation_result,
                    )
                    if config.save_predictions:
                        train_result = evaluate_model(
                            model=model,
                            target_workspace=protein_train_workspace,
                            data_loader=train_loader,
                            accumulation_steps=config.batch_size,
                            device=device,
                        )
                        save_prediction_tables(
                            output_dir=fold_output_dir,
                            train_predictions=train_result.predictions,
                            validation_predictions=validation_result.predictions,
                            test_predictions=test_result.predictions,
                        )
                else:
                    early_stop_counter += 1

                scheduler.step()
                if early_stop_counter >= config.early_stop_patience:
                    LOGGER.info(
                        "Early stopping triggered for outer fold %d / inner fold %d at epoch %d.",
                        outer_fold,
                        inner_fold,
                        epoch,
                    )
                    break

            best_model = build_model(
                input_dim=input_dim,
                num_proteins=prepared_inputs.protein_targets.shape[0],
                num_functions=prepared_inputs.function_targets.shape[0],
                num_attention_heads=config.model_num_attention_heads,
                device=device,
                dropout=config.model_dropout,
            )
            checkpoint_state = torch.load(best_checkpoint_path, map_location=device)
            best_model.load_state_dict(checkpoint_state["model_state_dict"])

            train_result = evaluate_model(
                model=best_model,
                target_workspace=protein_train_workspace,
                data_loader=train_loader,
                accumulation_steps=config.batch_size,
                device=device,
            )
            validation_result = evaluate_model(
                model=best_model,
                target_workspace=protein_validation_workspace,
                data_loader=validation_loader,
                accumulation_steps=config.batch_size,
                device=device,
            )
            test_result = evaluate_model(
                model=best_model,
                target_workspace=protein_test_workspace,
                data_loader=test_loader,
                accumulation_steps=config.batch_size,
                device=device,
            )

            if config.save_predictions:
                save_prediction_tables(
                    output_dir=fold_output_dir,
                    train_predictions=train_result.predictions,
                    validation_predictions=validation_result.predictions,
                    test_predictions=test_result.predictions,
                )

            summary_records.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "best_epoch": best_epoch,
                    "train_corr": train_result.mean_correlation,
                    "train_loss": train_result.mean_loss,
                    "validation_corr": validation_result.mean_correlation,
                    "validation_loss": validation_result.mean_loss,
                    "test_corr": test_result.mean_correlation,
                    "test_loss": test_result.mean_loss,
                }
            )

            LOGGER.info(
                "Completed outer fold %d / inner fold %d | best_epoch=%d | train_corr=%.6f | val_corr=%.6f | test_corr=%.6f",
                outer_fold,
                inner_fold,
                best_epoch,
                train_result.mean_correlation,
                validation_result.mean_correlation,
                test_result.mean_correlation,
            )

    summary_table = pd.DataFrame(summary_records)
    summary_table.to_csv(run_dir / "nested_cv_summary.csv", index=False)
    LOGGER.info("Training finished. Fold summary saved to %s", run_dir / "nested_cv_summary.csv")
