"""Genera explicaciones SHAP para el mejor modelo entrenado y guarda
gráficos resumen + explicaciones individuales.

Requiere el archivo best_model.pkl generado por preprocess_and_train.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "best_model.pkl"
OUT = ROOT / "outputs" / "shap"

# Crear carpeta de salida si no existe
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 300


def main():
    # Verificar que exista el modelo entrenado
    if not BEST.exists():
        print("No se encontró best_model.pkl. Ejecuta preprocess_and_train.py primero.")
        return

    # Cargar payload con modelo y metadatos
    payload = joblib.load(BEST)
    model = payload.get("modelo")
    feature_cols = payload.get("feature_columns", [])

    # Buscar dataset con churn ya construido
    shop = ROOT / "shopping_trends_churn.csv"
    if not shop.exists():
        print("No se encontró shopping_trends_churn.csv.")
        return

    # Leer dataset ignorando errores de codificación
    df = pd.read_csv(shop, encoding_errors="ignore")

    # Separar columnas numéricas
    num_cols = [
        c for c in ["previous_purchases", "purchase_amount_usd"]
        if c in df.columns
    ]

    # Detectar columnas categóricas
    cat_cols = [
        c for c in df.columns
        if c not in num_cols + ["churn", "customer_id"]
    ]

    # Limpiar columnas numéricas
    X_num = (
        df[num_cols].fillna(0)
        if num_cols else pd.DataFrame()
    )

    # Limpiar columnas categóricas
    X_cat = (
        df[cat_cols].fillna("missing").astype(str)
        if cat_cols else pd.DataFrame()
    )

    # Unir variables numéricas y categóricas
    X_df = (
        pd.concat([X_num, X_cat], axis=1)
        if not X_num.empty or not X_cat.empty
        else df
    )

    # Tomar una muestra aleatoria para SHAP
    sample_n = min(SAMPLE_SIZE, len(X_df))
    X_sample = X_df.sample(sample_n, random_state=RANDOM_STATE)

    # Transformar datos usando el preprocesador del pipeline
    X_proc = model.named_steps["pre"].transform(X_sample)

    # Intentar obtener nombres reales de features transformadas
    try:
        feature_names = model.named_steps["pre"].get_feature_names_out()

    except Exception:
        feature_names = [
            f"feature_{i}"
            for i in range(X_proc.shape[1])
        ]

    # Convertir matriz procesada a DataFrame
    X_proc_df = pd.DataFrame(
        X_proc,
        columns=feature_names,
        index=X_sample.index,
    )

    # Importar shap solo cuando sea necesario
    try:
        import shap

    except Exception:
        print("La librería shap no está instalada.")
        return

    # Crear explicador SHAP usando el clasificador del pipeline
    explainer = shap.Explainer(
        model.named_steps["clf"],
        X_proc_df,
    )

    # Calcular valores SHAP
    shap_values = explainer(X_proc_df)

    plot_values = shap_values

    # Ajustar dimensiones para clasificación binaria
    if getattr(shap_values, "values", np.array([])).ndim == 3:
        plot_values = shap_values[:, :, 1]

    # Generar gráfico resumen de importancia de variables
    try:
        shap.summary_plot(
            plot_values,
            X_proc_df,
            show=False,
            max_display=15,
        )

        plt.savefig(
            OUT / "summary.png",
            bbox_inches="tight",
            dpi=150,
        )

        plt.close()

        print(f"Resumen SHAP guardado en: {OUT / 'summary.png'}")

    except Exception as e:
        print("Error guardando gráfico resumen SHAP:", e)

    # Generar gráficos individuales para las primeras 3 filas
    for i in range(min(3, X_proc_df.shape[0])):
        try:
            shap.plots.waterfall(
                plot_values[i],
                max_display=12,
                show=False,
            )

            plt.savefig(
                OUT / f"waterfall_{i}.png",
                bbox_inches="tight",
                dpi=150,
            )

            plt.close()

        except Exception:
            pass


# Ejecutar script directamente
if __name__ == "__main__":
    main()