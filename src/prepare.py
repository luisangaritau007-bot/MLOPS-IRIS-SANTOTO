"""Carga Iris, controla calidad y genera los archivos train.csv y test.csv."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.validate_data import FEATURE_NAMES, ROOT_DIR, validate_rows

DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "processed" / "test.csv"


def prepare_dataset() -> tuple[Path, Path]:
    start = time.perf_counter()
    print("Cargando dataset Iris...")
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    frame["species"] = iris.target.astype(int)

    records = frame.to_dict(orient="records")
    errors = validate_rows(records)
    if errors:
        raise ValueError("El dataset Iris no superó el contrato:\n" + "\n".join(errors[:20]))

    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.20,
        random_state=42,
        stratify=frame["species"],
    )
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    test_frame.to_csv(TEST_PATH, index=False, encoding="utf-8")

    print("Preparación de datos finalizada.")
    print(f"Archivo de entrenamiento: {TRAIN_PATH}")
    print(f"Archivo de prueba: {TEST_PATH}")
    print(f"Registros de entrenamiento: {len(train_frame)}")
    print(f"Registros de prueba: {len(test_frame)}")
    print(f"Etapa completada correctamente en {time.perf_counter() - start:.2f} segundos.")
    return TRAIN_PATH, TEST_PATH


if __name__ == "__main__":
    prepare_dataset()
