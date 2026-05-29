"""Genera un CSV y un gráfico comparativo de métricas de modelos
usando outputs/train/evaluation.json generado por preprocess_and_train.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent

# Archivo con metricas de evaluacion generadas en entrenamiento
EVAL = ROOT / "outputs" / "train" / "evaluation.json"

# Carpeta de salida para CSV y gráficos
OUT = ROOT / "outputs" / "train"

# Crear carpeta si no existe
OUT.mkdir(parents=True, exist_ok=True)


def main():

    # Verificar que exista el archivo de evaluacion
    if not EVAL.exists():
        print(
            "No se encontró evaluation.json. "
            "Ejecuta preprocess_and_train.py primero."
        )
        return

    # Cargar métricas desde JSON
    data = json.loads(EVAL.read_text())

    # Convertir diccionario a DataFrame
    df = pd.DataFrame.from_dict(
        data,
        orient="index",
    )

    # Guardar comparación de metricas en CSV
    csv_out = OUT / "model_comparison.csv"

    df.to_csv(csv_out)

    print(f"CSV guardado: {csv_out}")

    # Crear grafico comparativo de ROC-AUC
    plt.figure(figsize=(6, 4))

    df["roc_auc"].plot(
        kind="bar",
        color=["#185FA5", "#E24B4A", "#BA7517"],
    )

    # Configuración visual del grafico
    plt.title("ROC-AUC por modelo")
    plt.ylabel("ROC-AUC")
    plt.ylim(0, 1)

    # Ajustar margenes automaticamente
    plt.tight_layout()

    # Ruta de salida del grafico PNG
    png_out = OUT / "roc_auc_comparison.png"

    plt.savefig(
        png_out,
        dpi=150,
    )

    # Cerrar figura para liberar memoria
    plt.close()

    print(f"PNG guardado: {png_out}")


# Ejecutar script directamente
if __name__ == "__main__":
    main()