# MLOPS-IRIS-SANTOTO

**Actividad 2.1 – Estandarización de Pipelines y Control de Calidad de Datos**  
**Dataset:** Iris (150 observaciones, cuatro variables predictoras y tres clases).  
**Modelo:** Random Forest Classifier.  
**Entorno de referencia:** Python 3.13 de 64 bits.

## 1. Objetivo

Implementar un pipeline reproducible que prepare los datos Iris, valide la calidad de los datos frente a un contrato JSON, entrene y evalúe un clasificador Random Forest, registre el experimento con MLflow y ejecute diez pruebas automatizadas. Todo el flujo se inicia con un solo comando.

## 2. Estructura del proyecto

```text
MLOPS-IRIS-SANTOTO/
├── data/
│   ├── train.csv                  # generado por el pipeline
│   └── processed/
│       └── test.csv               # generado por el pipeline
├── mlruns/                        # artefactos MLflow
├── models/
│   ├── model.pkl                  # modelo entrenado
│   └── metrics.json               # métricas de evaluación
├── src/
│   ├── __init__.py
│   ├── prepare.py                 # carga Iris y separa entrenamiento/prueba
│   ├── train.py                   # Random Forest y tracking MLflow
│   └── validate_data.py           # reglas de calidad con biblioteca estándar
├── tests/
│   └── test_model.py              # diez pruebas automatizadas
├── data_contract.json             # contrato oficial de datos
├── preparar_ui.py                 # captura datos y realiza inferencia
├── README.md
├── requirements.txt
├── romper_datos.py                # prueba deliberada de entrada inválida
└── run_pipeline.py                # orquestador de las cuatro etapas
```

Durante la ejecución también se crea `mlflow.db`, que contiene el tracking local de los experimentos. El entorno `.venv/` y la caché `.pytest_cache/` son locales y no se publican en GitHub.

## 3. Dataset y contrato de datos

El conjunto Iris se carga desde `sklearn.datasets.load_iris`. Cada observación contiene cuatro mediciones en centímetros:

| Variable | Descripción | Rango del conjunto Iris |
|---|---|---:|
| `sepal_length` | Longitud del sépalo | 4.3 a 7.9 |
| `sepal_width` | Ancho del sépalo | 2.0 a 4.4 |
| `petal_length` | Longitud del pétalo | 1.0 a 6.9 |
| `petal_width` | Ancho del pétalo | 0.1 a 2.5 |
| `species` | Clase objetivo | 0, 1 o 2 |

La variable objetivo se codifica como `0 = setosa`, `1 = versicolor` y `2 = virginica`. El archivo `data_contract.json` es la referencia oficial para la validación y la futura inferencia. Incluye tipo, obligatoriedad, unidad, rango permitido y ejemplos de entrada válida e inválida.

Los rangos son los mínimos y máximos observados en la versión clásica del conjunto Iris usada por Scikit-learn. Son reglas de calidad definidas para esta práctica, no límites biológicos universales.

## 4. Requisitos

- Windows 10/11 o sistema operativo compatible.
- Python 3.13 de 64 bits.
- Acceso a Internet durante la instalación de dependencias.

Las versiones solicitadas para reproducibilidad están fijadas en `requirements.txt`:

```text
mlflow==3.17.2
scikit-learn==1.9.1
pandas==3.0.5
joblib==1.6.0
pytest==9.2.0
```

## 5. Instalación desde cero (Windows PowerShell)

Abra Visual Studio Code en la carpeta raíz `MLOPS-IRIS-SANTOTO` y abra una terminal PowerShell.

### 5.1 Crear el entorno virtual

```powershell
py -3.13 -m venv .venv
```

### 5.2 Activar el entorno

```powershell
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución del script, habilite la política solo para la terminal actual y vuelva a activarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

### 5.3 Verificar Python e instalar dependencias

```powershell
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Compruebe que aparece el prefijo `(.venv)` en la terminal y que Python reporta versión 3.13.x.

## 6. Ejecución automatizada con un comando

Desde la carpeta raíz, ejecute:

```powershell
python run_pipeline.py
```

El orquestador ejecuta las siguientes etapas en orden:

1. **ETAPA 1 – Preparación de datos:** carga los 150 registros Iris, valida las filas originales, divide el conjunto de manera estratificada con 80 % para entrenamiento y 20 % para prueba (`random_state=42`) y guarda `data/train.csv` (120 registros) y `data/processed/test.csv` (30 registros).
2. **ETAPA 2 – Validación de calidad:** valida ambos CSV mediante las reglas definidas en el contrato. El proceso se detiene cuando encuentra columnas incorrectas, tipos incompatibles, valores nulos, valores no finitos o mediciones fuera de rango.
3. **ETAPA 3 – Entrenamiento y registro con MLflow:** entrena `RandomForestClassifier` con 100 árboles y semilla 42, calcula Accuracy, F1-score, matriz de confusión e informe de clasificación. Guarda `models/model.pkl` y `models/metrics.json`. MLflow registra el experimento `Proyecto_Final_Grado`, los parámetros, las métricas y los artefactos en `mlflow.db` y `mlruns/`.
4. **ETAPA 4 – Pruebas automatizadas:** ejecuta `tests/test_model.py`. Las diez pruebas verifican archivos, estructura, ausencia de nulos, dominio de la clase, carga del modelo, desempeño y ejemplos del contrato. Si alguna falla, el pipeline termina con error.

Al final se muestra `PIPELINE FINALIZADO CORRECTAMENTE` cuando las cuatro etapas finalizan sin errores.

## 7. Validación de datos sin librerías externas

El módulo `src/validate_data.py` usa exclusivamente módulos de la biblioteca estándar (`json`, `csv`, `math`, `pathlib` y `typing`) para validar entradas. Comprueba:

- Los cuatro campos obligatorios y que no existan columnas adicionales.
- Ausencia de valores nulos.
- Tipo numérico (`int` o `float`), sin aceptar booleanos como mediciones.
- Valores finitos; rechaza `NaN` e infinitos.
- Rangos mínimo y máximo por característica.
- Dominio de `species`, que debe ser un entero entre 0 y 2.

Pandas y Scikit-learn se usan para preparar el conjunto y entrenar/evaluar el modelo; no se usan para implementar las reglas del validador.

## 8. Ejecución de módulos por separado

Preparar únicamente los CSV:

```powershell
python -m src.prepare
```

Demostrar el rechazo de datos dañados deliberadamente:

```powershell
python romper_datos.py
```

Realizar una predicción interactiva (después de entrenar):

```powershell
python preparar_ui.py
```

Ejecutar las diez pruebas después de generar CSV, modelo y métricas:

```powershell
python -m pytest tests/test_model.py -v
```

## 9. Consultar experimentos en MLflow

El entrenamiento crea el experimento `Proyecto_Final_Grado` y almacena el seguimiento en `mlflow.db`, con artefactos en `mlruns/`. Para abrir la interfaz local:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra en el navegador la dirección mostrada por MLflow (normalmente `http://127.0.0.1:5000`).

### 9.1 Nota de serialización segura de MLflow

MLflow registra el modelo con el formato `skops`. En Random Forest, SKOPS puede marcar el tipo interno `sklearn.tree._tree.Tree` como no confiable. Como el modelo se entrena en este mismo pipeline y se conoce su origen, `src/train.py` autoriza explícitamente solo ese tipo mediante `skops_trusted_types=["sklearn.tree._tree.Tree"]`. No se deben añadir otros tipos a la lista sin revisar su procedencia y necesidad. Esta configuración evita cambiar a formatos basados en pickle/cloudpickle, que requieren precaución al cargar modelos de origen no confiable.

Si aparece una advertencia de Git indicando que el ejecutable no está en `PATH`, es un aviso independiente del registro del modelo: MLflow no podrá añadir metadatos del repositorio Git, pero el seguimiento local con SQLite puede funcionar. Para disponer de esos metadatos, instale Git for Windows, agréguelo al `PATH` y reinicie Visual Studio Code.

## 10. Archivos generados

| Archivo | Contenido |
|---|---|
| `data/train.csv` | 120 registros de entrenamiento |
| `data/processed/test.csv` | 30 registros de prueba |
| `models/model.pkl` | Clasificador Random Forest serializado con Joblib |
| `models/metrics.json` | Accuracy, F1-score, matriz e informe de clasificación |
| `mlflow.db` | Base de datos local de MLflow |
| `mlruns/` | Artefactos y registros del experimento |

Con la partición y configuración indicadas, el resultado de referencia es Accuracy = `0.9000` y F1-score ponderado = `0.8997`. Las métricas definitivas son las que se escriban en `models/metrics.json` al ejecutar el proyecto en el entorno local.

## 11. Publicación en GitHub y buenas prácticas

1. Publique `src/`, `tests/`, `data_contract.json`, `README.md`, `requirements.txt`, `preparar_ui.py`, `romper_datos.py` y `run_pipeline.py`.
2. No publique `.venv/`, `.pytest_cache/`, `__pycache__/`, `mlflow.db`, `mlruns/`, CSV derivados ni el modelo entrenado; estos artefactos se recrean al ejecutar el pipeline.
3. Mantenga actualizado el contrato, las pruebas y la documentación cuando cambien las variables o sus rangos.
4. No incluya información privada ni datos ajenos al conjunto Iris.
5. Para reproducir la ejecución, use Python 3.13 de 64 bits e instale las versiones exactas de `requirements.txt`.

## 12. Referencia del dataset

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics, 7*(2), 179–188.

El conjunto se carga desde la función `load_iris` de Scikit-learn.
