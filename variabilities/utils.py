import pandas as pd
import numpy as np


def calculate_wpv(df: pd.DataFrame, subject_column: str, features: list) -> pd.Series:
    return df.groupby(subject_column)[features].std().mean(axis=0)


def calculate_bpv(df: pd.DataFrame, subject_column: str, features: list) -> pd.Series:
    return df.groupby(subject_column)[features].mean().std(axis=0)


def propagate_error_ratio(wpv, bpv, err_wpv, err_bpv):
    ioi = (wpv / bpv).round(4)
    err_ioi = ioi * np.sqrt((err_wpv / wpv) ** 2 + (err_bpv / bpv) ** 2)
    return ioi, err_ioi


def cv(series):
    mean = series.mean()
    std = series.std()
    return std / mean if mean != 0 else np.nan


def calculate_wpcv(df: pd.DataFrame, subject_column: str, features: list) -> pd.Series:
    return df.groupby(subject_column)[features].agg(cv).mean(axis=0)


def calculate_bpcv(df: pd.DataFrame, subject_column: str, features: list) -> pd.Series:
    return df.groupby(subject_column)[features].mean().agg(cv)


def bootstrap_ci(
    df: pd.DataFrame,
    subject_column: str,
    features: list,
    bpv_func,
    wpv_func,
    n_boot: int = 1000,
    random_state: int = 0,
    ci: tuple[float, float] = (0.025, 0.975),
):
    rng = np.random.default_rng(random_state)
    subjects = df[subject_column].unique()
    n_subjects = len(subjects)

    # Cache per-subject data once (speeds up a lot)
    groups = {s: g for s, g in df.groupby(subject_column)}

    wpv_results = []
    bpv_results = []
    ii_results = []

    for i in range(n_boot):
        if  (i % 50 == 0):
            print(f"{i} / {n_boot}")

        sampled_subjects = rng.choice(subjects, size=n_subjects, replace=True)
        boot_df = pd.concat([groups[s] for s in sampled_subjects], ignore_index=True)

        wpv = wpv_func(boot_df, subject_column, features)  # Series (features,)
        bpv = bpv_func(boot_df, subject_column, features)  # Series (features,)
        ii = wpv / bpv  # Series (features,)

        wpv_results.append(wpv)
        bpv_results.append(bpv)
        ii_results.append(ii)

    wpv_results = pd.DataFrame(wpv_results)
    bpv_results = pd.DataFrame(bpv_results)
    ii_results = pd.DataFrame(ii_results)

    low, high = ci
    wpv_ci = wpv_results.quantile([low, high])
    bpv_ci = bpv_results.quantile([low, high])
    ii_ci = ii_results.quantile([low, high])

    # Nice row labels
    wpv_ci.index = ["low", "high"]
    bpv_ci.index = ["low", "high"]
    ii_ci.index = ["low", "high"]

    return wpv_ci, bpv_ci, ii_ci
