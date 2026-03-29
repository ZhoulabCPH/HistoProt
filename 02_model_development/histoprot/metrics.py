from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def safe_pearson_correlation(prediction_values: np.ndarray, target_values: np.ndarray) -> float:
    if prediction_values.size < 2 or target_values.size < 2:
        return np.nan
    if np.allclose(prediction_values, prediction_values[0]) or np.allclose(target_values, target_values[0]):
        return np.nan
    return pearsonr(prediction_values, target_values)[0]


def compute_mean_feature_correlation(
    target_workspace: pd.DataFrame,
    prediction_workspace: pd.DataFrame,
) -> float:
    correlations = []
    for feature_id in prediction_workspace.index:
        if feature_id not in target_workspace.index:
            continue
        target_values = target_workspace.loc[feature_id, :].to_numpy(dtype=float)
        prediction_values = prediction_workspace.loc[feature_id, :].to_numpy(dtype=float)
        correlations.append(safe_pearson_correlation(prediction_values, target_values))

    correlations = np.asarray(correlations, dtype=float)
    if np.isnan(correlations).all():
        return float("nan")
    return float(np.nanmean(correlations))
