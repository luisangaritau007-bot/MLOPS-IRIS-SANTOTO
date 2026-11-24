"""Diez pruebas automatizadas alineadas con la estructura de la práctica MLOps."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.validate_data import ROOT_DIR, load_contract, validate_record

TRAIN_PATH = ROOT_DIR / "data" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]


def test_processed_files_exist():
    assert TRAIN_PATH.is_file()
    assert TEST_PATH.is_file()


def test_datasets_are_not_empty():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    assert not train_df.empty
    assert not test_df.empty


def test_datasets_columns():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    assert train_df.columns.tolist() == EXPECTED_COLUMNS
    assert test_df.columns.tolist() == EXPECTED_COLUMNS


def test_no_null_values():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    assert not train_df.isnull().values.any()
    assert not test_df.isnull().values.any()


def test_target_domain():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    allowed = {0, 1, 2}
    assert set(train_df["species"].unique()).issubset(allowed)
    assert set(test_df["species"].unique()).issubset(allowed)


def test_model_file_exists():
    assert MODEL_PATH.is_file()
    assert METRICS_PATH.is_file()


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, RandomForestClassifier)


def test_model_accuracy():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    x_test = test_df.drop(columns=["species"])
    y_test = test_df["species"].astype(int)
    measured_accuracy = float(model.score(x_test, y_test))
    assert measured_accuracy >= 0.80
    assert abs(measured_accuracy - metrics["accuracy"]) < 1e-12
    assert metrics["train_records"] == len(train_df) == 120
    assert metrics["test_records"] == len(test_df) == 30
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_contract_valid_example_passes():
    contract = load_contract()
    valid_example = contract["examples"]["valid_input"]
    assert validate_record(valid_example) == []


def test_contract_invalid_example_is_rejected():
    contract = load_contract()
    invalid_example = contract["examples"]["invalid_input"]
    errors = validate_record(invalid_example)
    assert errors
    assert any("nulo" in error for error in errors)
    assert any("máximo" in error for error in errors)
