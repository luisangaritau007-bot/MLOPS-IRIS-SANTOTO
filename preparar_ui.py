"""Entrada interactiva para probar una inferencia con el contrato Iris."""
from __future__ import annotations

import joblib

from src.train import MODEL_PATH
from src.validate_data import FEATURE_NAMES, load_contract, validate_record


def main() -> None:
    contract = load_contract()
    print("\nCLASIFICADOR IRIS - MEDIDAS EN CENTÍMETROS")
    record = {}
    for feature, rule in contract["input_features"].items():
        while True:
            raw = input(f"{feature} ({rule['min']} a {rule['max']}): ").strip()
            try:
                record[feature] = float(raw)
                break
            except ValueError:
                print("Ingrese un valor numérico, por ejemplo 5.1.")

    errors = validate_record(record)
    if errors:
        print("Entrada rechazada por el contrato:")
        for error in errors:
            print(f" - {error}")
        return
    if not MODEL_PATH.exists():
        print("No existe el modelo. Ejecute primero: python run_pipeline.py")
        return

    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict([[record[name] for name in FEATURE_NAMES]])[0])
    label = contract["target_variable"]["labels"][str(prediction)]
    print(f"\nPredicción: Iris {label} (código {prediction}).")


if __name__ == "__main__":
    main()
