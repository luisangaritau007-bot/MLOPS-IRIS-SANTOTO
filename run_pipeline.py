"""Orquestador de ejecución única del pipeline MLOps Iris."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from src.prepare import prepare_dataset
from src.train import train_model
from src.validate_data import validate_csv_file

ROOT_DIR = Path(__file__).resolve().parent


def run_data_validation(paths: tuple[Path, Path]) -> None:
    """Aplica las reglas del contrato a los CSV antes del entrenamiento."""
    for path in paths:
        print(f"Validando: {path}", flush=True)
        errors = validate_csv_file(path)
        if errors:
            raise ValueError(
                f"El archivo {path} no superó el control de calidad:\n"
                + "\n".join(errors[:20])
            )
        print(f"[OK] Datos válidos: {path}", flush=True)


def run_tests() -> None:
    """Ejecuta las diez pruebas automatizadas después de generar el modelo."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_model.py", "-v"],
        cwd=ROOT_DIR,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Las pruebas automatizadas fallaron. Revise el reporte anterior.")


def main() -> None:
    total_start = time.perf_counter()
    print("PIPELINE MLOPS OPEN SOURCE - SANTOTO", flush=True)
    print("Estandarización y control de calidad de datos", flush=True)
    print("=" * 76, flush=True)

    print("\nETAPA 1: Preparación de datos", flush=True)
    print("=" * 76, flush=True)
    train_path, test_path = prepare_dataset()

    print("\nETAPA 2: Validación de calidad de datos", flush=True)
    print("=" * 76, flush=True)
    run_data_validation((train_path, test_path))
    print("Los datos fueron preparados y validados.", flush=True)

    print("\nETAPA 3: Entrenamiento y registro con MLflow", flush=True)
    print("=" * 76, flush=True)
    metrics = train_model(train_path, test_path)
    print("El modelo fue entrenado y almacenado.", flush=True)

    print("\nETAPA 4: Pruebas automatizadas", flush=True)
    print("=" * 76, flush=True)
    run_tests()
    print("Las pruebas automatizadas fueron superadas.", flush=True)

    elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 76, flush=True)
    print("PIPELINE FINALIZADO CORRECTAMENTE", flush=True)
    print("=" * 76, flush=True)
    print("Los datos fueron preparados y validados.", flush=True)
    print("El modelo fue entrenado y almacenado.", flush=True)
    print("Las pruebas automatizadas fueron superadas.", flush=True)
    print(f"Accuracy: {metrics['accuracy']:.4f} | F1-score: {metrics['f1_score']:.4f}", flush=True)
    print(f"Etapa completada correctamente en {elapsed:.2f} segundos.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
