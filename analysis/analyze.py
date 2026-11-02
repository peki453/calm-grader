import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Put the six prediction CSV files in the same folder as this script.
# The script automatically finds files containing "_predictions.csv".
INPUT_GLOB = "*_predictions.csv"

OUTPUT_DIR = BASE_DIR / "analysis_results"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
INDIVIDUAL_FIGURE_DIR = FIGURE_DIR / "individual"
GENERAL_FIGURE_DIR = FIGURE_DIR / "general"

# Selective grading coverage grid.
COVERAGES = np.arange(0.50, 1.001, 0.05)

# Reliability diagram.
ECE_BINS = 10

# Bootstrap confidence intervals.
# Set to 0 if you want to disable bootstrap CIs.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

# Primary uncertainty measure for risk-coverage figures.
PRIMARY_UNCERTAINTY = "entropy"

UNCERTAINTY_COLUMNS = [
    "entropy",
    "max_uncertainty",
    "margin_uncertainty",
    "variance",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_name(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_")


def infer_dataset_from_filename(path):
    name = path.name.lower()

    if "digiklausur" in name:
        return "DigiKlausur"

    if "mohler" in name:
        return "Mohler"

    return path.stem


def infer_provider_from_filename(path):
    name = path.name.lower()

    if "openai" in name:
        return "OpenAI"

    if "grok" in name:
        return "xAI"

    if "claude" in name:
        return "Anthropic"

    return "Unknown"


def rmse(y_true, y_pred):
    return math.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def parse_probabilities(value):
    if isinstance(value, dict):
        obj = value
    else:
        obj = json.loads(str(value))

    return {
        float(score): float(probability)
        for score, probability in obj.items()
    }


def extract_score_support(probability_dicts):
    values = set()

    for probs in probability_dicts:
        values.update(
            float(score)
            for score in probs.keys()
        )

    return sorted(values)


def align_gold_to_support(y_true, support):
    """
    Map a continuous/averaged gold score to the nearest score supported
    by the model. This is used ONLY for categorical metrics such as
    exact accuracy, Macro-F1, Cohen's kappa, QWK, ECE correctness,
    AUROC/AUPRC error detection, and confusion matrices.

    Raw MAE/RMSE always use the original human score.
    """
    support_array = np.asarray(
        support,
        dtype=float
    )

    aligned = []

    for value in np.asarray(
        y_true,
        dtype=float
    ):
        distance = np.abs(
            support_array - value
        )

        minimum = distance.min()

        candidates = support_array[
            np.isclose(
                distance,
                minimum
            )
        ]

        # Deterministic tie handling:
        # choose the candidate closest to the center of the support,
        # then the lower score if still tied.
        center = (
            support_array.min()
            + support_array.max()
        ) / 2

        candidates = sorted(
            candidates,
            key=lambda x: (
                abs(x - center),
                x
            )
        )

        aligned.append(
            float(candidates[0])
        )

    return np.asarray(
        aligned,
        dtype=float
    )



def to_class_codes(values, support):
    """
    Convert supported ordinal score values (e.g. 0, 0.5, 1, ...)
    to integer class IDs required by scikit-learn classification metrics.

    This changes only the representation, not the ordering.
    """
    support = [float(x) for x in support]

    mapping = {
        float(score): index
        for index, score in enumerate(support)
    }

    codes = []

    for value in np.asarray(values, dtype=float):
        matched = None

        for score, index in mapping.items():
            if np.isclose(value, score):
                matched = index
                break

        if matched is None:
            raise ValueError(
                f"Score {value} is not in supported score set {support}"
            )

        codes.append(matched)

    return np.asarray(codes, dtype=int)

def soft_target_for_score(
    human_score,
    support
):
    """
    Represent a potentially averaged human score as a probability
    distribution over adjacent supported ordinal classes.

    Example for support [..., 3.5, 4.0, ...]:
        gold 3.625 -> 0.75 mass at 3.5 and 0.25 at 4.0.

    For DigiKlausur, where gold labels are already 0/1/2,
    this reduces to an ordinary one-hot target.
    """
    support = np.asarray(
        support,
        dtype=float
    )

    target = np.zeros(
        len(support),
        dtype=float
    )

    value = float(
        human_score
    )

    if value <= support[0]:
        target[0] = 1.0
        return target

    if value >= support[-1]:
        target[-1] = 1.0
        return target

    exact = np.where(
        np.isclose(
            support,
            value
        )
    )[0]

    if len(exact):
        target[
            exact[0]
        ] = 1.0
        return target

    upper_idx = int(
        np.searchsorted(
            support,
            value,
            side="right"
        )
    )

    lower_idx = (
        upper_idx - 1
    )

    lower = support[
        lower_idx
    ]

    upper = support[
        upper_idx
    ]

    upper_weight = (
        value - lower
    ) / (
        upper - lower
    )

    lower_weight = (
        1.0 - upper_weight
    )

    target[
        lower_idx
    ] = lower_weight

    target[
        upper_idx
    ] = upper_weight

    return target


def probability_matrix(
    probability_dicts,
    support
):
    matrix = []

    for probs in probability_dicts:
        matrix.append(
            [
                float(
                    probs.get(
                        float(score),
                        0.0
                    )
                )
                for score in support
            ]
        )

    return np.asarray(
        matrix,
        dtype=float
    )


# ============================================================
# DATA LOADING + DEDUPLICATION
# ============================================================

def load_prediction_file(path):
    df = pd.read_csv(
        path
    )

    required = [
        "response_id",
        "item_id",
        "human_score",
        "model",
        "predicted_score",
        "expected_score",
        "probabilities",
        "max_probability",
        "entropy",
        "max_uncertainty",
        "margin_uncertainty",
        "variance",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            + ", ".join(
                missing
            )
        )

    dataset = infer_dataset_from_filename(
        path
    )

    if "dataset" not in df.columns:
        df["dataset"] = dataset
    else:
        df["dataset"] = (
            df["dataset"]
            .astype("object")
            .where(
                df["dataset"].notna(),
                dataset
            )
            .replace("", dataset)
            .astype(str)
        )

    provider = infer_provider_from_filename(
        path
    )

    if "provider" not in df.columns:
        df["provider"] = provider
    else:
        df["provider"] = (
            df["provider"]
            .astype("object")
            .where(
                df["provider"].notna(),
                provider
            )
            .replace("", provider)
            .astype(str)
        )

    if "status" not in df.columns:
        df["status"] = "ok"

    df["_source_file"] = path.name
    df["_row_order"] = np.arange(
        len(df)
    )

    # Prefer successful rows if the same response appears multiple times.
    df["_status_rank"] = (
        df["status"]
        .astype(str)
        .str.lower()
        .eq("ok")
        .astype(int)
    )

    original_rows = len(df)

    duplicate_rows = int(
        df["response_id"]
        .duplicated(
            keep=False
        )
        .sum()
    )

    # Sort so successful + latest row is kept.
    df = (
        df.sort_values(
            [
                "response_id",
                "_status_rank",
                "_row_order",
            ]
        )
        .drop_duplicates(
            subset=[
                "response_id"
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    unique_rows = len(df)

    success_mask = (
        df["status"]
        .astype(str)
        .str.lower()
        .eq("ok")
    )

    successful = df[
        success_mask
    ].copy()

    numeric_columns = [
        "human_score",
        "predicted_score",
        "expected_score",
        "absolute_error_map",
        "absolute_error_expected",
        "max_probability",
        "entropy",
        "max_uncertainty",
        "margin_uncertainty",
        "variance",
    ]

    for column in numeric_columns:
        if column in successful.columns:
            successful[
                column
            ] = pd.to_numeric(
                successful[
                    column
                ],
                errors="coerce"
            )

    successful = successful.dropna(
        subset=[
            "human_score",
            "predicted_score",
            "expected_score",
            "probabilities",
        ]
    ).copy()

    info = {
        "file":
            path.name,

        "dataset":
            str(
                successful[
                    "dataset"
                ].iloc[0]
            ) if len(successful) else dataset,

        "provider":
            str(
                successful[
                    "provider"
                ].iloc[0]
            ) if len(successful) else provider,

        "model":
            str(
                successful[
                    "model"
                ].iloc[0]
            ) if len(successful) else "Unknown",

        "original_rows":
            original_rows,

        "unique_response_ids":
            unique_rows,

        "duplicate_rows_detected":
            duplicate_rows,

        "successful_predictions":
            len(successful),

        "failed_or_invalid_unique_rows":
            unique_rows
            - len(successful),
    }

    return successful, info


# ============================================================
# CALIBRATION
# ============================================================

def calibration_metrics(
    y_true,
    y_aligned,
    y_pred,
    probability_dicts,
    support,
    bins=10
):
    probabilities = probability_matrix(
        probability_dicts,
        support
    )

    max_probability = probabilities.max(
        axis=1
    )

    # ECE uses categorical correctness after alignment
    # to the model-supported score scale.
    y_pred_codes = to_class_codes(
        y_pred,
        support
    )

    y_aligned_codes = to_class_codes(
        y_aligned,
        support
    )

    correctness = (
        y_pred_codes
        == y_aligned_codes
    ).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )

    ece = 0.0
    calibration_rows = []

    for i in range(
        bins
    ):
        lower = edges[i]
        upper = edges[i + 1]

        if i == 0:
            mask = (
                (max_probability >= lower)
                & (max_probability <= upper)
            )
        else:
            mask = (
                (max_probability > lower)
                & (max_probability <= upper)
            )

        if not np.any(
            mask
        ):
            continue

        observed_accuracy = float(
            correctness[
                mask
            ].mean()
        )

        mean_confidence = float(
            max_probability[
                mask
            ].mean()
        )

        count = int(
            mask.sum()
        )

        weight = (
            count
            / len(
                max_probability
            )
        )

        ece += (
            weight
            * abs(
                observed_accuracy
                - mean_confidence
            )
        )

        calibration_rows.append(
            {
                "bin_lower":
                    lower,

                "bin_upper":
                    upper,

                "n":
                    count,

                "mean_confidence":
                    mean_confidence,

                "observed_accuracy":
                    observed_accuracy,

                "calibration_gap":
                    mean_confidence
                    - observed_accuracy,
            }
        )

    # Soft-target Brier and NLL.
    # This is valid for both exact labels and Mohler averaged labels.
    brier_values = []
    nll_values = []

    for i, human_score in enumerate(
        y_true
    ):
        target = soft_target_for_score(
            human_score,
            support
        )

        prediction = probabilities[
            i
        ]

        brier_values.append(
            np.sum(
                (
                    prediction
                    - target
                ) ** 2
            )
        )

        clipped = np.clip(
            prediction,
            1e-12,
            1.0
        )

        nll_values.append(
            -np.sum(
                target
                * np.log(
                    clipped
                )
            )
        )

    return {
        "ece_aligned":
            float(
                ece
            ),

        "soft_brier":
            float(
                np.mean(
                    brier_values
                )
            ),

        "soft_nll":
            float(
                np.mean(
                    nll_values
                )
            ),

        "mean_confidence":
            float(
                np.mean(
                    max_probability
                )
            ),

        "confidence_minus_accuracy":
            float(
                np.mean(
                    max_probability
                )
                - np.mean(
                    correctness
                )
            ),
    }, pd.DataFrame(
        calibration_rows
    )


# ============================================================
# UNCERTAINTY / ERROR DETECTION
# ============================================================

def uncertainty_metrics(
    df,
    y_true,
    y_aligned,
    y_pred
):
    absolute_error = np.abs(
        np.asarray(
            y_pred,
            dtype=float
        )
        - np.asarray(
            y_true,
            dtype=float
        )
    )

    support = sorted(
        set(
            np.asarray(
                y_aligned,
                dtype=float
            )
        )
        | set(
            np.asarray(
                y_pred,
                dtype=float
            )
        )
    )

    y_aligned_codes = to_class_codes(
        y_aligned,
        support
    )

    y_pred_codes = to_class_codes(
        y_pred,
        support
    )

    classification_error = (
        y_pred_codes
        != y_aligned_codes
    ).astype(int)

    rows = []

    for column in UNCERTAINTY_COLUMNS:
        values = (
            df[
                column
            ]
            .astype(float)
            .to_numpy()
        )

        rho, p_value = spearmanr(
            values,
            absolute_error
        )

        try:
            auroc = roc_auc_score(
                classification_error,
                values
            )
        except ValueError:
            auroc = np.nan

        try:
            auprc = average_precision_score(
                classification_error,
                values
            )
        except ValueError:
            auprc = np.nan

        rows.append(
            {
                "uncertainty_measure":
                    column,

                "spearman_rho_vs_raw_absolute_error":
                    rho,

                "spearman_p":
                    p_value,

                "error_detection_auroc":
                    auroc,

                "error_detection_auprc":
                    auprc,

                "error_prevalence":
                    classification_error.mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SELECTIVE GRADING
# ============================================================

def selective_metrics(
    df,
    support,
    uncertainty_column=PRIMARY_UNCERTAINTY
):
    ordered = (
        df.sort_values(
            uncertainty_column,
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    rows = []

    for coverage in COVERAGES:
        n = max(
            1,
            int(
                round(
                    len(
                        ordered
                    )
                    * coverage
                )
            )
        )

        subset = ordered.iloc[
            :n
        ]

        y_true = (
            subset[
                "human_score"
            ]
            .astype(float)
            .to_numpy()
        )

        y_pred = (
            subset[
                "predicted_score"
            ]
            .astype(float)
            .to_numpy()
        )

        y_expected = (
            subset[
                "expected_score"
            ]
            .astype(float)
            .to_numpy()
        )

        y_aligned = align_gold_to_support(
            y_true,
            support
        )

        y_aligned_codes = to_class_codes(
            y_aligned,
            support
        )

        y_pred_codes = to_class_codes(
            y_pred,
            support
        )

        score_range = (
            max(
                support
            )
            - min(
                support
            )
        )

        rows.append(
            {
                "coverage":
                    float(
                        coverage
                    ),

                "n_accepted":
                    n,

                "n_referred":
                    len(
                        ordered
                    )
                    - n,

                "review_rate":
                    1.0
                    - coverage,

                "accuracy_aligned":
                    accuracy_score(
                        y_aligned_codes,
                        y_pred_codes
                    ),

                "qwk_aligned":
                    cohen_kappa_score(
                        y_aligned_codes,
                        y_pred_codes,
                        weights="quadratic"
                    ),

                "mae_map_raw":
                    mean_absolute_error(
                        y_true,
                        y_pred
                    ),

                "mae_expected_raw":
                    mean_absolute_error(
                        y_true,
                        y_expected
                    ),

                "normalized_mae_expected":
                    mean_absolute_error(
                        y_true,
                        y_expected
                    ) / score_range
                    if score_range > 0
                    else np.nan,

                "mean_uncertainty":
                    subset[
                        uncertainty_column
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def area_under_risk_coverage(
    selective_df,
    risk_column="normalized_mae_expected"
):
    x = selective_df[
        "coverage"
    ].to_numpy()

    y = selective_df[
        risk_column
    ].to_numpy()

    return float(
        np.trapezoid(
            y,
            x
        )
        / (
            x.max()
            - x.min()
        )
    )


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(
    y_true,
    y_aligned,
    y_pred,
    y_expected,
    uncertainty,
    n_boot=1000,
    seed=42
):
    if n_boot <= 0:
        return {}

    rng = np.random.default_rng(
        seed
    )

    n = len(
        y_true
    )

    values = {
        "accuracy_aligned":
            [],

        "qwk_aligned":
            [],

        "mae_expected_raw":
            [],

        "entropy_error_auroc":
            [],
    }

    support = sorted(
        set(
            np.asarray(
                y_aligned,
                dtype=float
            )
        )
        | set(
            np.asarray(
                y_pred,
                dtype=float
            )
        )
    )

    y_aligned_codes_all = to_class_codes(
        y_aligned,
        support
    )

    y_pred_codes_all = to_class_codes(
        y_pred,
        support
    )

    error_binary = (
        y_pred_codes_all
        != y_aligned_codes_all
    ).astype(int)

    for _ in range(
        n_boot
    ):
        idx = rng.integers(
            0,
            n,
            n
        )

        yt = y_true[
            idx
        ]

        ya = y_aligned[
            idx
        ]

        yp = y_pred[
            idx
        ]

        ya_codes = y_aligned_codes_all[
            idx
        ]

        yp_codes = y_pred_codes_all[
            idx
        ]

        ye = y_expected[
            idx
        ]

        uu = uncertainty[
            idx
        ]

        values[
            "accuracy_aligned"
        ].append(
            accuracy_score(
                ya_codes,
                yp_codes
            )
        )

        try:
            qwk = cohen_kappa_score(
                ya_codes,
                yp_codes,
                weights="quadratic"
            )
        except Exception:
            qwk = np.nan

        values[
            "qwk_aligned"
        ].append(
            qwk
        )

        values[
            "mae_expected_raw"
        ].append(
            mean_absolute_error(
                yt,
                ye
            )
        )

        eb = error_binary[
            idx
        ]

        if len(
            np.unique(
                eb
            )
        ) == 2:
            try:
                auc = roc_auc_score(
                    eb,
                    uu
                )
            except Exception:
                auc = np.nan
        else:
            auc = np.nan

        values[
            "entropy_error_auroc"
        ].append(
            auc
        )

    result = {}

    for metric, samples in values.items():
        arr = np.asarray(
            samples,
            dtype=float
        )

        arr = arr[
            np.isfinite(
                arr
            )
        ]

        if len(
            arr
        ) == 0:
            result[
                metric + "_ci_low"
            ] = np.nan

            result[
                metric + "_ci_high"
            ] = np.nan

        else:
            result[
                metric + "_ci_low"
            ] = float(
                np.percentile(
                    arr,
                    2.5
                )
            )

            result[
                metric + "_ci_high"
            ] = float(
                np.percentile(
                    arr,
                    97.5
                )
            )

    return result


# ============================================================
# RUN-LEVEL ANALYSIS
# ============================================================

def analyze_run(
    df,
    file_info
):
    probability_dicts = [
        parse_probabilities(
            value
        )
        for value
        in df[
            "probabilities"
        ]
    ]

    support = extract_score_support(
        probability_dicts
    )

    y_true = (
        df[
            "human_score"
        ]
        .astype(float)
        .to_numpy()
    )

    y_pred = (
        df[
            "predicted_score"
        ]
        .astype(float)
        .to_numpy()
    )

    y_expected = (
        df[
            "expected_score"
        ]
        .astype(float)
        .to_numpy()
    )

    y_aligned = align_gold_to_support(
        y_true,
        support
    )

    score_range = (
        max(
            support
        )
        - min(
            support
        )
    )

    raw_exact_accuracy = np.mean(
        np.isclose(
            y_true,
            y_pred
        )
    )

    y_aligned_codes = to_class_codes(
        y_aligned,
        support
    )

    y_pred_codes = to_class_codes(
        y_pred,
        support
    )

    aligned_accuracy = accuracy_score(
        y_aligned_codes,
        y_pred_codes
    )

    aligned_macro_f1 = f1_score(
        y_aligned_codes,
        y_pred_codes,
        average="macro",
        zero_division=0
    )

    aligned_kappa = cohen_kappa_score(
        y_aligned_codes,
        y_pred_codes
    )

    aligned_qwk = cohen_kappa_score(
        y_aligned_codes,
        y_pred_codes,
        weights="quadratic"
    )

    mae_map_raw = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse_map_raw = rmse(
        y_true,
        y_pred
    )

    mae_expected_raw = mean_absolute_error(
        y_true,
        y_expected
    )

    rmse_expected_raw = rmse(
        y_true,
        y_expected
    )

    calibration, calibration_bins = (
        calibration_metrics(
            y_true,
            y_aligned,
            y_pred,
            probability_dicts,
            support,
            bins=ECE_BINS
        )
    )

    uncertainty_df = (
        uncertainty_metrics(
            df,
            y_true,
            y_aligned,
            y_pred
        )
    )

    entropy_row = (
        uncertainty_df[
            uncertainty_df[
                "uncertainty_measure"
            ]
            == "entropy"
        ]
        .iloc[0]
    )

    selective_df = (
        selective_metrics(
            df,
            support,
            uncertainty_column=PRIMARY_UNCERTAINTY
        )
    )

    aurc = area_under_risk_coverage(
        selective_df
    )

    entropy_values = (
        df[
            "entropy"
        ]
        .astype(float)
        .to_numpy()
    )

    bootstrap = bootstrap_ci(
        y_true,
        y_aligned,
        y_pred,
        y_expected,
        entropy_values,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED
    )

    summary = {
        **file_info,

        "score_support":
            json.dumps(
                support
            ),

        "score_min":
            min(
                support
            ),

        "score_max":
            max(
                support
            ),

        "score_range":
            score_range,

        "n_items":
            df[
                "item_id"
            ].nunique(),

        "raw_exact_accuracy":
            raw_exact_accuracy,

        "accuracy_aligned":
            aligned_accuracy,

        "macro_f1_aligned":
            aligned_macro_f1,

        "cohen_kappa_aligned":
            aligned_kappa,

        "qwk_aligned":
            aligned_qwk,

        "mae_map_raw":
            mae_map_raw,

        "rmse_map_raw":
            rmse_map_raw,

        "mae_expected_raw":
            mae_expected_raw,

        "rmse_expected_raw":
            rmse_expected_raw,

        "normalized_mae_map":
            mae_map_raw
            / score_range
            if score_range > 0
            else np.nan,

        "normalized_mae_expected":
            mae_expected_raw
            / score_range
            if score_range > 0
            else np.nan,

        **calibration,

        "mean_entropy":
            df[
                "entropy"
            ].mean(),

        "mean_max_uncertainty":
            df[
                "max_uncertainty"
            ].mean(),

        "mean_margin_uncertainty":
            df[
                "margin_uncertainty"
            ].mean(),

        "mean_variance":
            df[
                "variance"
            ].mean(),

        "entropy_spearman_rho":
            entropy_row[
                "spearman_rho_vs_raw_absolute_error"
            ],

        "entropy_spearman_p":
            entropy_row[
                "spearman_p"
            ],

        "entropy_error_detection_auroc":
            entropy_row[
                "error_detection_auroc"
            ],

        "entropy_error_detection_auprc":
            entropy_row[
                "error_detection_auprc"
            ],

        "classification_error_prevalence":
            entropy_row[
                "error_prevalence"
            ],

        "aurc_normalized_expected_mae":
            aurc,

        **bootstrap,
    }

    # Add identifiers to supporting tables.
    for frame in [
        calibration_bins,
        uncertainty_df,
        selective_df,
    ]:
        frame.insert(
            0,
            "dataset",
            file_info[
                "dataset"
            ]
        )

        frame.insert(
            1,
            "provider",
            file_info[
                "provider"
            ]
        )

        frame.insert(
            2,
            "model",
            file_info[
                "model"
            ]
        )

    # Class-wise metrics on aligned gold labels.
    labels = list(
        range(
            len(
                support
            )
        )
    )

    precision, recall, f1, support_counts = (
        precision_recall_fscore_support(
            y_aligned_codes,
            y_pred_codes,
            labels=labels,
            zero_division=0
        )
    )

    class_rows = []

    for (
        label,
        p,
        r,
        f,
        n
    ) in zip(
        labels,
        precision,
        recall,
        f1,
        support_counts
    ):
        class_rows.append(
            {
                "dataset":
                    file_info[
                        "dataset"
                    ],

                "provider":
                    file_info[
                        "provider"
                    ],

                "model":
                    file_info[
                        "model"
                    ],

                "score":
                    support[
                        int(label)
                    ],

                "precision":
                    p,

                "recall":
                    r,

                "f1":
                    f,

                "support":
                    n,
            }
        )

    class_df = pd.DataFrame(
        class_rows
    )

    # Item/question-level statistics.
    item_rows = []

    working = df.copy()

    working[
        "_gold_aligned"
    ] = y_aligned

    working[
        "_abs_error_map_raw"
    ] = np.abs(
        y_pred
        - y_true
    )

    working[
        "_abs_error_expected_raw"
    ] = np.abs(
        y_expected
        - y_true
    )

    working[
        "_correct_aligned"
    ] = (
        y_pred
        == y_aligned
    ).astype(int)

    for item_id, group in working.groupby(
        "item_id"
    ):
        item_rows.append(
            {
                "dataset":
                    file_info[
                        "dataset"
                    ],

                "provider":
                    file_info[
                        "provider"
                    ],

                "model":
                    file_info[
                        "model"
                    ],

                "item_id":
                    item_id,

                "n":
                    len(
                        group
                    ),

                "accuracy_aligned":
                    group[
                        "_correct_aligned"
                    ].mean(),

                "mae_map_raw":
                    group[
                        "_abs_error_map_raw"
                    ].mean(),

                "mae_expected_raw":
                    group[
                        "_abs_error_expected_raw"
                    ].mean(),

                "mean_entropy":
                    group[
                        "entropy"
                    ].mean(),

                "mean_confidence":
                    group[
                        "max_probability"
                    ].mean(),
            }
        )

    item_df = pd.DataFrame(
        item_rows
    )

    return {
        "summary":
            summary,

        "calibration":
            calibration_bins,

        "uncertainty":
            uncertainty_df,

        "selective":
            selective_df,

        "class_metrics":
            class_df,

        "item_metrics":
            item_df,

        "support":
            support,

        "y_true":
            y_true,

        "y_aligned":
            y_aligned,

        "y_pred":
            y_pred,

        "y_expected":
            y_expected,

        "probability_dicts":
            probability_dicts,

        "df":
            df.copy(),
    }


# ============================================================
# GENERAL PLOTS
# ============================================================

def save_general_bar_plot(
    summary_df,
    metric,
    ylabel,
    filename,
    title
):
    datasets = list(
        summary_df[
            "dataset"
        ].drop_duplicates()
    )

    models = list(
        summary_df[
            "model"
        ].drop_duplicates()
    )

    x = np.arange(
        len(
            models
        )
    )

    width = (
        0.8
        / max(
            1,
            len(
                datasets
            )
        )
    )

    plt.figure(
        figsize=(
            max(
                8,
                len(
                    models
                )
                * 2.2
            ),
            5.5
        )
    )

    for i, dataset in enumerate(
        datasets
    ):
        sub = (
            summary_df[
                summary_df[
                    "dataset"
                ]
                == dataset
            ]
            .set_index(
                "model"
            )
            .reindex(
                models
            )
        )

        offset = (
            i
            - (
                len(
                    datasets
                )
                - 1
            ) / 2
        ) * width

        plt.bar(
            x + offset,
            sub[
                metric
            ].to_numpy(),
            width=width,
            label=dataset
        )

    plt.xticks(
        x,
        models,
        rotation=15,
        ha="right"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        GENERAL_FIGURE_DIR
        / f"{filename}.png",
        dpi=300
    )

    plt.savefig(
        GENERAL_FIGURE_DIR
        / f"{filename}.pdf"
    )

    plt.close()


def plot_confidence_vs_accuracy(
    summary_df
):
    plt.figure(
        figsize=(7, 6)
    )

    for dataset, group in summary_df.groupby(
        "dataset"
    ):
        plt.scatter(
            group[
                "mean_confidence"
            ],
            group[
                "accuracy_aligned"
            ],
            label=dataset,
            s=70
        )

        for _, row in group.iterrows():
            plt.annotate(
                row[
                    "model"
                ],
                (
                    row[
                        "mean_confidence"
                    ],
                    row[
                        "accuracy_aligned"
                    ]
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel(
        "Mean confidence"
    )

    plt.ylabel(
        "Aligned exact accuracy"
    )

    plt.title(
        "Confidence versus observed accuracy"
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        GENERAL_FIGURE_DIR
        / "05_confidence_vs_accuracy.png",
        dpi=300
    )

    plt.savefig(
        GENERAL_FIGURE_DIR
        / "05_confidence_vs_accuracy.pdf"
    )

    plt.close()


def plot_general_risk_coverage(
    selective_all
):
    for dataset, group in selective_all.groupby(
        "dataset"
    ):
        plt.figure(
            figsize=(7, 5)
        )

        for model, model_group in group.groupby(
            "model"
        ):
            model_group = model_group.sort_values(
                "coverage"
            )

            plt.plot(
                model_group[
                    "coverage"
                ]
                * 100,
                model_group[
                    "normalized_mae_expected"
                ],
                marker="o",
                label=model
            )

        plt.xlabel(
            "Automatic grading coverage (%)"
        )

        plt.ylabel(
            "Normalized expected-score MAE"
        )

        plt.title(
            f"Selective grading risk–coverage: {dataset}"
        )

        plt.grid(
            alpha=0.25
        )

        plt.legend()

        plt.tight_layout()

        name = safe_name(
            dataset
        )

        plt.savefig(
            GENERAL_FIGURE_DIR
            / f"06_risk_coverage_{name}.png",
            dpi=300
        )

        plt.savefig(
            GENERAL_FIGURE_DIR
            / f"06_risk_coverage_{name}.pdf"
        )

        plt.close()


# ============================================================
# INDIVIDUAL PLOTS
# ============================================================

def plot_individual_reliability(
    analysis,
    dataset,
    model
):
    calibration = analysis[
        "calibration"
    ]

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    if len(
        calibration
    ):
        plt.plot(
            calibration[
                "mean_confidence"
            ],
            calibration[
                "observed_accuracy"
            ],
            marker="o",
            label="Model"
        )

    plt.xlabel(
        "Mean predicted confidence"
    )

    plt.ylabel(
        "Observed accuracy"
    )

    plt.title(
        f"Reliability diagram\n{dataset} — {model}"
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    prefix = (
        safe_name(
            dataset
        )
        + "__"
        + safe_name(
            model
        )
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__reliability.png",
        dpi=300
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__reliability.pdf"
    )

    plt.close()


def plot_individual_confusion(
    analysis,
    dataset,
    model
):
    y_aligned = analysis[
        "y_aligned"
    ]

    y_pred = analysis[
        "y_pred"
    ]

    support = analysis[
        "support"
    ]

    y_aligned_codes = to_class_codes(
        y_aligned,
        support
    )

    y_pred_codes = to_class_codes(
        y_pred,
        support
    )

    labels = list(
        range(
            len(
                support
            )
        )
    )

    cm = confusion_matrix(
        y_aligned_codes,
        y_pred_codes,
        labels=labels
    )

    plt.figure(
        figsize=(
            max(
                6,
                len(
                    labels
                )
                * 0.65
            ),
            max(
                5,
                len(
                    labels
                )
                * 0.55
            )
        )
    )

    image = plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.colorbar(
        image,
        label="Responses"
    )

    plt.xticks(
        range(
            len(
                labels
            )
        ),
        [
            f"{support[int(x)]:g}"
            for x in labels
        ],
        rotation=45
        if len(
            labels
        ) > 5
        else 0
    )

    plt.yticks(
        range(
            len(
                labels
            )
        ),
        [
            f"{x:g}"
            for x in labels
        ]
    )

    plt.xlabel(
        "Predicted score"
    )

    plt.ylabel(
        "Aligned human score"
    )

    plt.title(
        f"Confusion matrix\n{dataset} — {model}"
    )

    # Add counts only if matrix remains readable.
    if len(
        labels
    ) <= 11:
        for i in range(
            cm.shape[
                0
            ]
        ):
            for j in range(
                cm.shape[
                    1
                ]
            ):
                plt.text(
                    j,
                    i,
                    str(
                        cm[
                            i,
                            j
                        ]
                    ),
                    ha="center",
                    va="center",
                    fontsize=7
                    if len(
                        labels
                    ) > 5
                    else 10
                )

    plt.tight_layout()

    prefix = (
        safe_name(
            dataset
        )
        + "__"
        + safe_name(
            model
        )
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__confusion_matrix.png",
        dpi=300
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__confusion_matrix.pdf"
    )

    plt.close()


def plot_individual_uncertainty_box(
    analysis,
    dataset,
    model
):
    df = analysis[
        "df"
    ].copy()

    y_aligned = analysis[
        "y_aligned"
    ]

    y_pred = analysis[
        "y_pred"
    ]

    correct = (
        y_aligned
        == y_pred
    )

    correct_entropy = (
        df.loc[
            correct,
            "entropy"
        ]
        .astype(float)
        .to_numpy()
    )

    incorrect_entropy = (
        df.loc[
            ~correct,
            "entropy"
        ]
        .astype(float)
        .to_numpy()
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.boxplot(
        [
            correct_entropy,
            incorrect_entropy
        ],
        tick_labels=[
            "Correct",
            "Incorrect"
        ],
        showmeans=True
    )

    plt.ylabel(
        "Normalized entropy"
    )

    plt.title(
        f"Uncertainty by grading outcome\n{dataset} — {model}"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    prefix = (
        safe_name(
            dataset
        )
        + "__"
        + safe_name(
            model
        )
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__entropy_correct_incorrect.png",
        dpi=300
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__entropy_correct_incorrect.pdf"
    )

    plt.close()


def plot_individual_risk_coverage(
    analysis,
    dataset,
    model
):
    selective = analysis[
        "selective"
    ].sort_values(
        "coverage"
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        selective[
            "coverage"
        ]
        * 100,
        selective[
            "normalized_mae_expected"
        ],
        marker="o"
    )

    plt.xlabel(
        "Automatic grading coverage (%)"
    )

    plt.ylabel(
        "Normalized expected-score MAE"
    )

    plt.title(
        f"Risk–coverage curve\n{dataset} — {model}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    prefix = (
        safe_name(
            dataset
        )
        + "__"
        + safe_name(
            model
        )
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__risk_coverage.png",
        dpi=300
    )

    plt.savefig(
        INDIVIDUAL_FIGURE_DIR
        / f"{prefix}__risk_coverage.pdf"
    )

    plt.close()


# ============================================================
# PAIRWISE MODEL COMPARISONS
# ============================================================

def pairwise_model_comparisons(
    analyses
):
    rows = []

    by_dataset = {}

    for key, analysis in analyses.items():
        dataset, model = key

        by_dataset.setdefault(
            dataset,
            {}
        )[
            model
        ] = analysis

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    for dataset, models in by_dataset.items():
        model_names = sorted(
            models.keys()
        )

        for model_a, model_b in combinations(
            model_names,
            2
        ):
            a = models[
                model_a
            ]

            b = models[
                model_b
            ]

            da = a[
                "df"
            ][
                [
                    "response_id",
                    "human_score",
                    "predicted_score",
                    "expected_score",
                ]
            ].copy()

            db = b[
                "df"
            ][
                [
                    "response_id",
                    "human_score",
                    "predicted_score",
                    "expected_score",
                ]
            ].copy()

            merged = da.merge(
                db,
                on="response_id",
                suffixes=(
                    "_a",
                    "_b"
                )
            )

            if merged.empty:
                continue

            support_a = a[
                "support"
            ]

            support_b = b[
                "support"
            ]

            # They should have the same score support within a dataset.
            common_support = sorted(
                set(
                    support_a
                )
                | set(
                    support_b
                )
            )

            y_true = (
                merged[
                    "human_score_a"
                ]
                .astype(float)
                .to_numpy()
            )

            y_aligned = align_gold_to_support(
                y_true,
                common_support
            )

            pred_a = (
                merged[
                    "predicted_score_a"
                ]
                .astype(float)
                .to_numpy()
            )

            pred_b = (
                merged[
                    "predicted_score_b"
                ]
                .astype(float)
                .to_numpy()
            )

            y_aligned_codes = to_class_codes(
                y_aligned,
                common_support
            )

            pred_a_codes = to_class_codes(
                pred_a,
                common_support
            )

            pred_b_codes = to_class_codes(
                pred_b,
                common_support
            )

            expected_a = (
                merged[
                    "expected_score_a"
                ]
                .astype(float)
                .to_numpy()
            )

            expected_b = (
                merged[
                    "expected_score_b"
                ]
                .astype(float)
                .to_numpy()
            )

            correct_a = (
                pred_a_codes
                == y_aligned_codes
            )

            correct_b = (
                pred_b_codes
                == y_aligned_codes
            )

            mae_a = np.abs(
                expected_a
                - y_true
            )

            mae_b = np.abs(
                expected_b
                - y_true
            )

            # Paired bootstrap difference:
            # A - B, so negative MAE difference favors A.
            boot_acc = []
            boot_mae = []

            if BOOTSTRAP_N > 0:
                n = len(
                    merged
                )

                for _ in range(
                    BOOTSTRAP_N
                ):
                    idx = rng.integers(
                        0,
                        n,
                        n
                    )

                    boot_acc.append(
                        correct_a[
                            idx
                        ].mean()
                        - correct_b[
                            idx
                        ].mean()
                    )

                    boot_mae.append(
                        mae_a[
                            idx
                        ].mean()
                        - mae_b[
                            idx
                        ].mean()
                    )

                acc_low, acc_high = np.percentile(
                    boot_acc,
                    [
                        2.5,
                        97.5
                    ]
                )

                mae_low, mae_high = np.percentile(
                    boot_mae,
                    [
                        2.5,
                        97.5
                    ]
                )

            else:
                acc_low = np.nan
                acc_high = np.nan
                mae_low = np.nan
                mae_high = np.nan

            rows.append(
                {
                    "dataset":
                        dataset,

                    "model_a":
                        model_a,

                    "model_b":
                        model_b,

                    "n_common":
                        len(
                            merged
                        ),

                    "accuracy_a":
                        correct_a.mean(),

                    "accuracy_b":
                        correct_b.mean(),

                    "accuracy_difference_a_minus_b":
                        correct_a.mean()
                        - correct_b.mean(),

                    "accuracy_diff_ci_low":
                        acc_low,

                    "accuracy_diff_ci_high":
                        acc_high,

                    "expected_mae_a":
                        mae_a.mean(),

                    "expected_mae_b":
                        mae_b.mean(),

                    "expected_mae_difference_a_minus_b":
                        mae_a.mean()
                        - mae_b.mean(),

                    "expected_mae_diff_ci_low":
                        mae_low,

                    "expected_mae_diff_ci_high":
                        mae_high,

                    "a_correct_b_wrong":
                        int(
                            (
                                correct_a
                                & ~correct_b
                            ).sum()
                        ),

                    "b_correct_a_wrong":
                        int(
                            (
                                correct_b
                                & ~correct_a
                            ).sum()
                        ),

                    "both_correct":
                        int(
                            (
                                correct_a
                                & correct_b
                            ).sum()
                        ),

                    "both_wrong":
                        int(
                            (
                                ~correct_a
                                & ~correct_b
                            ).sum()
                        ),

                    "a_lower_expected_error":
                        int(
                            (
                                mae_a
                                < mae_b
                            ).sum()
                        ),

                    "b_lower_expected_error":
                        int(
                            (
                                mae_b
                                < mae_a
                            ).sum()
                        ),

                    "equal_expected_error":
                        int(
                            np.isclose(
                                mae_a,
                                mae_b
                            ).sum()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MODEL AGREEMENT
# ============================================================

def model_agreement_tables(
    analyses
):
    rows = []

    by_dataset = {}

    for (
        dataset,
        model
    ), analysis in analyses.items():
        by_dataset.setdefault(
            dataset,
            {}
        )[
            model
        ] = analysis

    for dataset, models in by_dataset.items():
        model_names = sorted(
            models.keys()
        )

        for model_a, model_b in combinations(
            model_names,
            2
        ):
            da = models[
                model_a
            ][
                "df"
            ][
                [
                    "response_id",
                    "predicted_score"
                ]
            ]

            db = models[
                model_b
            ][
                "df"
            ][
                [
                    "response_id",
                    "predicted_score"
                ]
            ]

            merged = da.merge(
                db,
                on="response_id",
                suffixes=(
                    "_a",
                    "_b"
                )
            )

            if merged.empty:
                continue

            pa = (
                merged[
                    "predicted_score_a"
                ]
                .astype(float)
                .to_numpy()
            )

            pb = (
                merged[
                    "predicted_score_b"
                ]
                .astype(float)
                .to_numpy()
            )

            agreement_support = sorted(
                set(
                    pa
                )
                | set(
                    pb
                )
            )

            pa_codes = to_class_codes(
                pa,
                agreement_support
            )

            pb_codes = to_class_codes(
                pb,
                agreement_support
            )

            rows.append(
                {
                    "dataset":
                        dataset,

                    "model_a":
                        model_a,

                    "model_b":
                        model_b,

                    "n_common":
                        len(
                            merged
                        ),

                    "exact_prediction_agreement":
                        np.mean(
                            pa
                            == pb
                        ),

                    "mean_absolute_prediction_difference":
                        np.mean(
                            np.abs(
                                pa
                                - pb
                            )
                        ),

                    "quadratic_weighted_kappa_between_models":
                        cohen_kappa_score(
                            pa_codes,
                            pb_codes,
                            weights="quadratic"
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# AUTOMATIC RESULTS INSIGHTS
# ============================================================

def create_insights(
    summary_df,
    selective_df,
    pairwise_df
):
    lines = []

    lines.append(
        "AUTOMATIC RESULTS NOTES"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        ""
    )

    lines.append(
        "Important methodological note:"
    )

    lines.append(
        "For categorical metrics on Mohler, averaged human scores are aligned "
        "to the nearest score supported by the model. Raw MAE/RMSE use the "
        "original averaged human scores. Brier score and NLL use soft ordinal "
        "targets obtained by interpolating between adjacent supported classes."
    )

    lines.append(
        ""
    )

    for dataset, group in summary_df.groupby(
        "dataset"
    ):
        lines.append(
            f"DATASET: {dataset}"
        )

        lines.append(
            "-" * 72
        )

        best_qwk = group.sort_values(
            "qwk_aligned",
            ascending=False
        ).iloc[0]

        best_mae = group.sort_values(
            "normalized_mae_expected",
            ascending=True
        ).iloc[0]

        best_cal = group.sort_values(
            "ece_aligned",
            ascending=True
        ).iloc[0]

        best_auc = group.sort_values(
            "entropy_error_detection_auroc",
            ascending=False
        ).iloc[0]

        lines.append(
            f"Best ordinal agreement: {best_qwk['model']} "
            f"(QWK={best_qwk['qwk_aligned']:.3f})."
        )

        lines.append(
            f"Lowest normalized expected-score MAE: {best_mae['model']} "
            f"({best_mae['normalized_mae_expected']:.3f})."
        )

        lines.append(
            f"Best raw calibration by ECE: {best_cal['model']} "
            f"(ECE={best_cal['ece_aligned']:.3f})."
        )

        lines.append(
            f"Best entropy-based error detection: {best_auc['model']} "
            f"(AUROC={best_auc['entropy_error_detection_auroc']:.3f}, "
            f"AUPRC={best_auc['entropy_error_detection_auprc']:.3f})."
        )

        lines.append(
            ""
        )

        for _, row in group.sort_values(
            "model"
        ).iterrows():
            lines.append(
                f"{row['model']}: accuracy={row['accuracy_aligned']:.3f}, "
                f"QWK={row['qwk_aligned']:.3f}, "
                f"normalized expected MAE={row['normalized_mae_expected']:.3f}, "
                f"ECE={row['ece_aligned']:.3f}, "
                f"mean confidence={row['mean_confidence']:.3f}, "
                f"entropy-error AUROC={row['entropy_error_detection_auroc']:.3f}."
            )

        lines.append(
            ""
        )

        sel = selective_df[
            selective_df[
                "dataset"
            ]
            == dataset
        ]

        for model, mgroup in sel.groupby(
            "model"
        ):
            full = mgroup.iloc[
                (
                    mgroup[
                        "coverage"
                    ]
                    - 1.0
                ).abs()
                .argsort()
                [:1]
            ].iloc[0]

            cov70 = mgroup.iloc[
                (
                    mgroup[
                        "coverage"
                    ]
                    - 0.70
                ).abs()
                .argsort()
                [:1]
            ].iloc[0]

            improvement = (
                full[
                    "normalized_mae_expected"
                ]
                - cov70[
                    "normalized_mae_expected"
                ]
            )

            lines.append(
                f"Selective grading ({model}): at ~70% coverage, normalized "
                f"expected-score MAE={cov70['normalized_mae_expected']:.3f} "
                f"versus {full['normalized_mae_expected']:.3f} at full coverage "
                f"(absolute reduction={improvement:.3f})."
            )

        lines.append(
            ""
        )

    if len(
        pairwise_df
    ):
        lines.append(
            "PAIRWISE COMPARISONS"
        )

        lines.append(
            "-" * 72
        )

        for _, row in pairwise_df.iterrows():
            lines.append(
                f"{row['dataset']}: {row['model_a']} vs {row['model_b']} — "
                f"accuracy difference A-B={row['accuracy_difference_a_minus_b']:.3f} "
                f"(95% bootstrap CI {row['accuracy_diff_ci_low']:.3f} to "
                f"{row['accuracy_diff_ci_high']:.3f}); expected-MAE difference "
                f"A-B={row['expected_mae_difference_a_minus_b']:.3f} "
                f"(95% bootstrap CI {row['expected_mae_diff_ci_low']:.3f} to "
                f"{row['expected_mae_diff_ci_high']:.3f})."
            )

    lines.append(
        ""
    )

    lines.append(
        "Suggested Results-section structure:"
    )

    lines.append(
        "1. Predictive performance across models and datasets."
    )

    lines.append(
        "2. Probabilistic calibration and over/under-confidence."
    )

    lines.append(
        "3. Relationship between uncertainty and grading error."
    )

    lines.append(
        "4. Selective grading / coverage-risk trade-off."
    )

    lines.append(
        "5. Per-question variation and model disagreement."
    )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    INDIVIDUAL_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GENERAL_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        BASE_DIR.glob(
            INPUT_GLOB
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No files matching {INPUT_GLOB} found in:\n"
            f"{BASE_DIR}"
        )

    print(
        f"Found {len(files)} prediction files."
    )

    analyses = {}

    summaries = []
    data_quality_rows = []
    calibration_frames = []
    uncertainty_frames = []
    selective_frames = []
    class_frames = []
    item_frames = []

    for path in files:
        print(
            f"\nAnalyzing: {path.name}"
        )

        df, info = load_prediction_file(
            path
        )

        data_quality_rows.append(
            info.copy()
        )

        if df.empty:
            print(
                "  No valid successful predictions; skipped."
            )
            continue

        analysis = analyze_run(
            df,
            info
        )

        key = (
            info[
                "dataset"
            ],
            info[
                "model"
            ]
        )

        analyses[
            key
        ] = analysis

        summaries.append(
            analysis[
                "summary"
            ]
        )

        calibration_frames.append(
            analysis[
                "calibration"
            ]
        )

        uncertainty_frames.append(
            analysis[
                "uncertainty"
            ]
        )

        selective_frames.append(
            analysis[
                "selective"
            ]
        )

        class_frames.append(
            analysis[
                "class_metrics"
            ]
        )

        item_frames.append(
            analysis[
                "item_metrics"
            ]
        )

        plot_individual_reliability(
            analysis,
            info[
                "dataset"
            ],
            info[
                "model"
            ]
        )

        plot_individual_confusion(
            analysis,
            info[
                "dataset"
            ],
            info[
                "model"
            ]
        )

        plot_individual_uncertainty_box(
            analysis,
            info[
                "dataset"
            ],
            info[
                "model"
            ]
        )

        plot_individual_risk_coverage(
            analysis,
            info[
                "dataset"
            ],
            info[
                "model"
            ]
        )

        print(
            f"  N={len(df)}, "
            f"QWK={analysis['summary']['qwk_aligned']:.3f}, "
            f"NMAE={analysis['summary']['normalized_mae_expected']:.3f}, "
            f"ECE={analysis['summary']['ece_aligned']:.3f}"
        )

    if not summaries:
        raise ValueError(
            "No analyzable runs were found."
        )

    summary_df = pd.DataFrame(
        summaries
    )

    data_quality_df = pd.DataFrame(
        data_quality_rows
    )

    calibration_all = pd.concat(
        calibration_frames,
        ignore_index=True
    )

    uncertainty_all = pd.concat(
        uncertainty_frames,
        ignore_index=True
    )

    selective_all = pd.concat(
        selective_frames,
        ignore_index=True
    )

    class_all = pd.concat(
        class_frames,
        ignore_index=True
    )

    item_all = pd.concat(
        item_frames,
        ignore_index=True
    )

    pairwise_df = pairwise_model_comparisons(
        analyses
    )

    agreement_df = model_agreement_tables(
        analyses
    )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    summary_df.to_csv(
        TABLE_DIR
        / "01_model_dataset_summary.csv",
        index=False
    )

    data_quality_df.to_csv(
        TABLE_DIR
        / "02_data_quality_and_duplicates.csv",
        index=False
    )

    uncertainty_all.to_csv(
        TABLE_DIR
        / "03_uncertainty_error_detection.csv",
        index=False
    )

    calibration_all.to_csv(
        TABLE_DIR
        / "04_calibration_bins.csv",
        index=False
    )

    selective_all.to_csv(
        TABLE_DIR
        / "05_selective_grading.csv",
        index=False
    )

    class_all.to_csv(
        TABLE_DIR
        / "06_class_metrics.csv",
        index=False
    )

    item_all.to_csv(
        TABLE_DIR
        / "07_item_question_metrics.csv",
        index=False
    )

    pairwise_df.to_csv(
        TABLE_DIR
        / "08_pairwise_model_comparisons.csv",
        index=False
    )

    agreement_df.to_csv(
        TABLE_DIR
        / "09_model_agreement.csv",
        index=False
    )

    # Compact paper table.
    paper_columns = [
        "dataset",
        "provider",
        "model",
        "successful_predictions",
        "accuracy_aligned",
        "macro_f1_aligned",
        "qwk_aligned",
        "mae_map_raw",
        "mae_expected_raw",
        "normalized_mae_expected",
        "ece_aligned",
        "soft_brier",
        "soft_nll",
        "mean_confidence",
        "entropy_spearman_rho",
        "entropy_error_detection_auroc",
        "entropy_error_detection_auprc",
        "aurc_normalized_expected_mae",
        "accuracy_aligned_ci_low",
        "accuracy_aligned_ci_high",
        "qwk_aligned_ci_low",
        "qwk_aligned_ci_high",
        "mae_expected_raw_ci_low",
        "mae_expected_raw_ci_high",
        "entropy_error_auroc_ci_low",
        "entropy_error_auroc_ci_high",
    ]

    paper_df = summary_df[
        [
            column
            for column in paper_columns
            if column in summary_df.columns
        ]
    ].copy()

    paper_df.to_csv(
        TABLE_DIR
        / "10_paper_ready_main_table.csv",
        index=False
    )

    # ========================================================
    # GENERAL FIGURES
    # ========================================================

    save_general_bar_plot(
        summary_df,
        metric="qwk_aligned",
        ylabel="Quadratic weighted kappa",
        filename="01_qwk_by_model_dataset",
        title="Ordinal grading agreement across models and datasets"
    )

    save_general_bar_plot(
        summary_df,
        metric="normalized_mae_expected",
        ylabel="Normalized expected-score MAE",
        filename="02_normalized_mae_by_model_dataset",
        title="Normalized grading error across models and datasets"
    )

    save_general_bar_plot(
        summary_df,
        metric="ece_aligned",
        ylabel="Expected calibration error",
        filename="03_ece_by_model_dataset",
        title="Calibration error across models and datasets"
    )

    save_general_bar_plot(
        summary_df,
        metric="entropy_error_detection_auroc",
        ylabel="AUROC",
        filename="04_uncertainty_auroc_by_model_dataset",
        title="Entropy-based grading-error detection"
    )

    plot_confidence_vs_accuracy(
        summary_df
    )

    plot_general_risk_coverage(
        selective_all
    )

    # ========================================================
    # INSIGHTS TEXT
    # ========================================================

    insights = create_insights(
        summary_df,
        selective_all,
        pairwise_df
    )

    insights_path = (
        OUTPUT_DIR
        / "results_insights.txt"
    )

    insights_path.write_text(
        insights,
        encoding="utf-8"
    )

    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        f"Main paper table:\n"
        f"{TABLE_DIR / '10_paper_ready_main_table.csv'}"
    )

    print(
        f"\nAutomatic results notes:\n"
        f"{insights_path}"
    )

    print(
        f"\nGeneral figures:\n"
        f"{GENERAL_FIGURE_DIR}"
    )

    print(
        f"\nIndividual figures:\n"
        f"{INDIVIDUAL_FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
