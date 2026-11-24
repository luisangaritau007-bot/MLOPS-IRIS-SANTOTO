"""Demuestra que el contrato rechaza nulos y valores fuera de rango."""
from __future__ import annotations

import json

from src.validate_data import load_contract, validate_record


def main() -> None:
    contract = load_contract()
    damaged = dict(contract["examples"]["valid_input"])
    damaged["sepal_length"] = None
    damaged["petal_width"] = 4.8
    print("Registro alterado deliberadamente:")
    print(json.dumps(damaged, indent=2, ensure_ascii=False))
    errors = validate_record(damaged)
    if not errors:
        raise SystemExit("ERROR: el validador aceptó un registro inválido.")
    print("\n[OK] El control de calidad rechazó la entrada:")
    for error in errors:
        print(f" - {error}")


if __name__ == "__main__":
    main()
