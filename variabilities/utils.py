import pandas as pd
import numpy as np

hematology_abbreviation = {
    "hemoglobin": "Hgb",
    "hematocrit": "Htc",
    "white_blood_cells": "WBC",
    "platelet_count": "Plt",
    "total_protein": "TP",
    "ast": "AST",
    "alt": "ALT",
    "ggt": "GGT",
    "lactate_dehydrogenase": "LDH",
    "albumin": "Alb",
    "sodium": "Na",
    "potassium": "K",
    "calcium": "Ca",
    "chloride": "Cl",
    "total_bilirubin": "Bi",
    "total_cholesterol": "Chol",
    "ldl_cholesterol": "LDL-C",
    "hdl_cholesterol": "HDL-C",
    "triglycerides": "TG",
    "creatinine": "Crea",
    "glucose": "Glu",
    "insulin": "Ins",
    "hba1c": "HbA1c",
    "tsh": "TSH",
    "crp": "CRP",
    "cea": "CEA",
    "egfr": "eGFR",
}

blood_parameters = [
    "hemoglobin",
    "hematocrit",
    "white_blood_cells",
    "platelet_count",
    "ast",
    "alt",
    "ggt",
    "lactate_dehydrogenase",
    "creatinine",
    "egfr",
    "sodium",
    "potassium",
    "calcium",
    "chloride",
    "total_cholesterol",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "total_bilirubin",
    "total_protein",
    "albumin",
    "glucose",
    "insulin",
    "hba1c",
    "tsh",
    "cea",
    "crp",
]


def calculate_wpv(
    df: pd.DataFrame, features: list, groupby_columns: list = ["subject_id"]
) -> pd.Series:
    return df.groupby(groupby_columns, observed=False)[features].std().mean()


def calculate_bpv(
    df: pd.DataFrame, features: list, groupby_columns: list = ["visit"]
) -> pd.Series:
    return df.groupby(groupby_columns, observed=False)[features].std().mean()


def calculate_wpv_uncertainty(
    df: pd.DataFrame, features: list, groupby_columns: list = ["subject_id"]
) -> pd.Series:
    return df.groupby(groupby_columns, observed=False)[features].std().std()


def calculate_bpv_uncertainty(
    df: pd.DataFrame, features: list, groupby_columns: list = ["visit"]
) -> pd.Series:
    return df.groupby(groupby_columns, observed=False)[features].std().std()


def propagate_error_ratio(wpv, bpv, err_wpv, err_bpv):
    ioi = (wpv / bpv).round(4)
    err_ioi = ioi * np.sqrt((err_wpv / wpv) ** 2 + (err_bpv / bpv) ** 2)
    return ioi, err_ioi
