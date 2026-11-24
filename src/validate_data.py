"""Control de calidad de datos Iris usando solo módulos de la biblioteca estándar."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT_DIR / "data_contract.json"
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAME = "species"


def load_contract(contract_path: str | Path = CONTRACT_PATH) -> dict[str, Any]:
    """Lee el contrato de datos en formato JSON."""
    with Path(contract_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_record(record: Any, contract_path: str | Path = CONTRACT_PATH) -> list[str]:
    """Valida un objeto de inferencia; una lista vacía significa que es válido."""
    contract = load_contract(contract_path)
    features = contract["input_features"]
    if not isinstance(record, dict):
        return ["La entrada debe ser un objeto JSON/diccionario."]

    errors: list[str] = []
    expected = set(features)
    received = set(record)
    for field in sorted(expected - received):
        errors.append(f"Falta el campo obligatorio '{field}'.")
    for field in sorted(received - expected):
        errors.append(f"El campo '{field}' no está definido en el contrato.")

    for field, rules in features.items():
        if field not in record:
            continue
        value = record[field]
        if value is None:
            errors.append(f"El campo '{field}' contiene un valor nulo.")
            continue
        # bool es subclase de int en Python, pero no es una medida aceptable.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"El campo '{field}' debe ser numérico (int o float).")
            continue
        if not math.isfinite(float(value)):
            errors.append(f"El campo '{field}' debe ser un número finito.")
            continue
        if value < rules["min"]:
            errors.append(f"El campo '{field}' está por debajo del mínimo permitido ({rules['min']}).")
        if value > rules["max"]:
            errors.append(f"El campo '{field}' supera el máximo permitido ({rules['max']}).")
    return errors


def validate_target(value: Any, contract_path: str | Path = CONTRACT_PATH) -> list[str]:
    """Valida la clase objetivo codificada como entero 0, 1 o 2."""
    allowed = load_contract(contract_path)["target_variable"]["allowed_values"]
    if isinstance(value, bool) or not isinstance(value, int):
        return ["La variable objetivo 'species' debe ser un número entero."]
    if value not in allowed:
        return [f"La variable objetivo 'species' debe pertenecer a {allowed}."]
    return []


def validate_rows(rows: list[dict[str, Any]], contract_path: str | Path = CONTRACT_PATH) -> list[str]:
    """Valida filas con cuatro características y la clase objetivo."""
    if not rows:
        return ["El conjunto de datos está vacío."]
    errors: list[str] = []
    expected_columns = set(FEATURE_NAMES + [TARGET_NAME])
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"Fila {index}: debe ser un diccionario.")
            continue
        missing = expected_columns - set(row)
        extra = set(row) - expected_columns
        for name in sorted(missing):
            errors.append(f"Fila {index}: falta la columna '{name}'.")
        for name in sorted(extra):
            errors.append(f"Fila {index}: columna no permitida '{name}'.")
        feature_record = {key: row[key] for key in FEATURE_NAMES if key in row}
        errors.extend(f"Fila {index}: {message}" for message in validate_record(feature_record, contract_path))
        if TARGET_NAME in row:
            errors.extend(f"Fila {index}: {message}" for message in validate_target(row[TARGET_NAME], contract_path))
    return errors


def _parse_csv_value(value: str | None, target: bool = False) -> Any:
    if value is None or value.strip() == "":
        return None
    try:
        if target:
            number = float(value)
            return int(number) if number.is_integer() else value
        return float(value)
    except (TypeError, ValueError):
        return value


def validate_csv_file(csv_path: str | Path, contract_path: str | Path = CONTRACT_PATH) -> list[str]:
    """Valida un CSV con csv/json/math; no depende de pandas ni NumPy."""
    path = Path(csv_path)
    if not path.exists():
        return [f"No existe el archivo: {path}"]
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        expected = FEATURE_NAMES + [TARGET_NAME]
        if reader.fieldnames != expected:
            return [f"Columnas incorrectas. Esperadas: {expected}; recibidas: {reader.fieldnames}."]
        rows = []
        for raw in reader:
            row = {name: _parse_csv_value(raw.get(name), target=(name == TARGET_NAME)) for name in expected}
            rows.append(row)
    return validate_rows(rows, contract_path)
