from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class HistoProtSpatialInference(nn.Module):
    """HistoProt inference model exposing both patch-level and slide-level outputs."""

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

    def forward_patch(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        return self.patch_protein_head(patch_embeddings)

    def forward_region(
        self,
        patch_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_logits = self.region_attention(patch_embeddings)
        attention_logits = attention_logits.transpose(1, 2)
        raw_attention = attention_logits
        attention_weights = F.softmax(attention_logits, dim=2)
        region_embedding = (attention_weights @ patch_embeddings).view(
            patch_embeddings.size(0), -1,
        )

        protein_predictions = self.region_protein_head(region_embedding)
        function_predictions = self.region_function_head(region_embedding)
        return protein_predictions, function_predictions, raw_attention

    def forward_slide(
        self,
        patch_embeddings: torch.Tensor,
        region_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.slide_attention(patch_embeddings)
        attention_logits = attention_logits.transpose(1, 2)
        attention_logits = attention_logits * region_attention
        attention_weights = F.softmax(attention_logits, dim=2)
        slide_embedding = (attention_weights @ patch_embeddings).view(
            patch_embeddings.size(0), -1,
        )

        protein_predictions = self.slide_protein_head(slide_embedding)
        function_predictions = self.slide_function_head(slide_embedding)
        return protein_predictions, function_predictions

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
            patch_region_assignments = patch_region_assignments.to(device=patch_features.device, dtype=torch.long)

        patch_embeddings = self.feature_projection(patch_features).unsqueeze(0)
        patch_level_predictions = self.forward_patch(patch_embeddings)

        unique_region_ids = torch.unique(patch_region_assignments)
        region_level_predictions: list[torch.Tensor] = []
        region_level_function_predictions: list[torch.Tensor] = []
        region_attention_maps: list[torch.Tensor] = []

        for region_id in unique_region_ids:
            region_indices = torch.where(patch_region_assignments == region_id)[0]
            region_proteins, region_functions, region_attention = self.forward_region(
                patch_embeddings[:, region_indices, :],
            )
            region_level_predictions.append(region_proteins)
            region_level_function_predictions.append(region_functions)
            region_attention_maps.append(region_attention)

        region_attention_tensor = torch.cat(region_attention_maps, dim=2)
        region_level_predictions_t = torch.cat(region_level_predictions)
        region_level_function_predictions_t = torch.cat(region_level_function_predictions)

        slide_protein_predictions, slide_function_predictions = self.forward_slide(
            patch_embeddings,
            region_attention_tensor,
        )

        return {
            "patch_level_predictions": patch_level_predictions.squeeze(0),
            "region_level_predictions": region_level_predictions_t,
            "region_level_function_predictions": region_level_function_predictions_t,
            "slide_protein_predictions": slide_protein_predictions.squeeze(0),
            "slide_function_predictions": slide_function_predictions.squeeze(0),
        }
