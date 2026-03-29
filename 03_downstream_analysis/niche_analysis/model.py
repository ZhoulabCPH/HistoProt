from __future__ import annotations

from typing import Any

import torch
from torch import nn


class HistoProtPatchEmbeddingInference(nn.Module):
    """Inference-time HistoProt wrapper exposing patch embeddings."""

    def __init__(
        self,
        input_dim: int = 512,
        num_proteins: int = 7715,
        num_functions: int = 50,
        num_attention_heads: int = 1,
        dropout: float = 0.25,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self.input_dim = input_dim
        self.embedding_dim = input_dim
        self.attention_hidden_dim = self.embedding_dim // 2
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.num_proteins = num_proteins
        self.num_functions = num_functions

        pooled_dim = self.embedding_dim * self.num_attention_heads

        self.feature_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.patch_protein_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim * 2, self.num_proteins),
        )

        self.region_attention = self._build_attention_module()
        self.region_protein_head = nn.Linear(pooled_dim, self.num_proteins)
        self.region_function_head = nn.Linear(pooled_dim, self.num_functions)

        self.slide_attention = self._build_attention_module()
        self.slide_protein_head = nn.Linear(pooled_dim, self.num_proteins)
        self.slide_function_head = nn.Linear(pooled_dim, self.num_functions)

        self._initialize_weights()

    def _build_attention_module(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.embedding_dim, self.attention_hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.attention_hidden_dim, self.num_attention_heads),
        )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_region_assignments: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if patch_features.dim() != 2:
            raise ValueError(
                f"Expected patch_features to be 2-D (num_patches, feature_dim), "
                f"got {patch_features.dim()}-D."
            )

        if not isinstance(patch_region_assignments, torch.Tensor):
            patch_region_assignments = torch.as_tensor(
                patch_region_assignments,
                dtype=torch.long,
                device=patch_features.device,
            )
        else:
            patch_region_assignments = patch_region_assignments.to(
                device=patch_features.device,
                dtype=torch.long,
            )

        if patch_region_assignments.dim() != 1:
            raise ValueError(
                "Expected patch_region_assignments to be 1-D with one region label per patch, "
                f"got shape {tuple(patch_region_assignments.shape)}."
            )
        if patch_region_assignments.numel() != patch_features.shape[0]:
            raise ValueError(
                "The number of patch region assignments does not match the number of patches. "
                f"Assignments: {patch_region_assignments.numel()}, patches: {patch_features.shape[0]}."
            )

        patch_embedding = self.feature_projection(patch_features)
        return {"patch_embedding": patch_embedding}
