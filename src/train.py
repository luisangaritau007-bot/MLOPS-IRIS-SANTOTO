"""Entrenamiento Random Forest, evaluación y registro del experimento en MLflow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.validate_data import FEATURE_NAMES, ROOT_DIR, validate_csv_file

TRAIN_PATH = ROOT_DIR / "data" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
MLRUNS_DIR = ROOT_DIR / "mlruns"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
EXPERIMENT_NAME = "Proyecto_Final_Grado"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def initialize_mlflow():
    """Inicializa MLflow y crea/reutiliza el experimento de la profesora."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print(
            "[ADVERTENCIA] MLflow no está disponible en este intérprete. "
            "El entrenamiento continuará sin tracking. Instale requirements.txt "
            "para habilitar el registro completo."
        )
        return None

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=MLRUNS_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def train_model(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> dict[str, Any]:
    """Valida CSV, entrena Random Forest, calcula métricas y guarda artefactos."""
    for dataset_path in (train_path, test_path):
        errors = validate_csv_file(dataset_path)
        if errors:
            raise ValueError(
                f"El archivo {dataset_path} no cumple el contrato:\n"
                + "\n".join(errors[:20])
            )

    print("Cargando datos procesados...", flush=True)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    x_train = train_df[FEATURE_NAMES]
    y_train = train_df["species"].astype(int)
    x_test = test_df[FEATURE_NAMES]
    y_test = test_df["species"].astype(int)

    mlflow = initialize_mlflow()

    print("Iniciando entrenamiento...", flush=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, average="weighted"))
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist()
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": "iris_random_forest_classifier",
        "algorithm": "RandomForestClassifier",
        "random_state": 42,
        "n_estimators": 100,
        "test_size": 0.20,
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix_labels": CLASS_NAMES,
        "confusion_matrix": matrix,
        "classification_report": report,
        "feature_importances": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, model.feature_importances_)
        },
        "model_file": "models/model.pkl",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mlflow_run_id = None
    if mlflow is not None:
        with mlflow.start_run(run_name="RandomForest_Iris") as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_params(
                {
                    "algorithm": "RandomForestClassifier",
                    "n_estimators": 100,
                    "random_state": 42,
                    "test_size": 0.20,
                    "features": ",".join(FEATURE_NAMES),
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "train_records": len(train_df),
                    "test_records": len(test_df),
                }
            )
            mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_file")
            # MLflow usa el formato seguro SKOPS por defecto. Este modelo Random Forest
            # se acaba de entrenar con scikit-learn dentro de este proyecto; se autoriza
            # únicamente el tipo interno que SKOPS identificó como necesario para
            # reconstruir los árboles del bosque. No ampliar esta lista sin revisar el tipo.
            mlflow.sklearn.log_model(
                model,
                name="random_forest_iris",
                serialization_format="skops",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

    print("Entrenamiento finalizado.", flush=True)
    print(f"Accuracy: {accuracy:.4f}", flush=True)
    print(f"F1-score: {f1:.4f}", flush=True)
    print(f"Modelo guardado en: {MODEL_PATH}", flush=True)
    if mlflow_run_id:
        print(f"MLflow Run ID: {mlflow_run_id}", flush=True)
    print(f"Métricas guardadas en: {METRICS_PATH}", flush=True)
    return metrics


if __name__ == "__main__":
    train_model()
