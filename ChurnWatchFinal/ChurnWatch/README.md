# ChurnWatch

Proyecto final de Aprendizaje de Maquina para detectar clientes de retail con posible riesgo de churn usando una variable proxy, modelos supervisados e interpretabilidad.

## Dataset principal

El dataset principal es `shopping_trends.csv`, con 3,900 registros. Como no contiene una columna real de abandono, se construye una variable proxy:

```text
churn = 1 si:
  frequency_of_purchases esta en {Annually, Every 3 Months, Quarterly}
  y previous_purchases <= percentil 25
```

En este dataset el percentil 25 de `previous_purchases` es 13. Con la regla corregida quedan:

- `churn = 0`: 3,448 clientes
- `churn = 1`: 452 clientes
- churn rate proxy: 11.59%

## Abrir en Visual Studio 2022

1. Abre Visual Studio 2022.
2. Selecciona `Open a local folder`.
3. Abre la carpeta `ChurnWatch`.
4. Crea o selecciona un entorno de Python.
5. Instala dependencias:

```bash
python -m pip install -r requirements.txt
```

## Flujo de ejecucion

Ejecuta los scripts desde la carpeta `ChurnWatch`:

```bash
python build_churn.py
python preprocess_and_train.py
python compute_shap.py
python app.py
```

La app Dash queda disponible en:

```text
http://localhost:8050
```

## Resultados actuales

El entrenamiento corregido compara tres modelos:

- Logistic Regression
- Random Forest
- XGBoost

Las metricas se guardan en:

- `outputs/train/evaluation.json`
- `outputs/train/model_comparison.csv`
- `outputs/train/target_summary.json`

El mejor modelo se guarda como `best_model.pkl`. La app carga ese archivo automaticamente; si no existe o falla, usa el modo heuristico como respaldo.

## Nota metodologica importante

Como `churn` es una variable proxy construida a partir de frecuencia de compra y compras previas, las metricas pueden salir muy altas cuando esas mismas variables entran al modelo. Esto se debe explicar en la presentacion: el proyecto no predice abandono real observado, sino riesgo de abandono definido por una regla de negocio basada en comportamiento.

## Archivos principales

- `build_churn.py`: construye `shopping_trends_churn.csv`.
- `preprocess_and_train.py`: entrena y evalua modelos.
- `compute_shap.py`: genera graficas SHAP.
- `app.py`: interfaz Dash para cargar clientes y ver riesgo.
- `run_eda.py`: analisis exploratorio de los datasets.
