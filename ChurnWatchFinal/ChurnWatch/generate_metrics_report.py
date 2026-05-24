"""Genera una tabla CSV y un PNG comparativo con las metricas de los modelos
usando outputs/train/evaluation.json generado por preprocess_and_train.py
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "outputs" / "train" / "evaluation.json"
OUT = ROOT / "outputs" / "train"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not EVAL.exists():
        print("No evaluation.json found. Ejecuta preprocess_and_train.py primero.")
        return
    data = json.loads(EVAL.read_text())
    df = pd.DataFrame.from_dict(data, orient="index")
    csv_out = OUT / "model_comparison.csv"
    df.to_csv(csv_out)
    print(f"CSV guardado: {csv_out}")

    # simple bar chart for AUC
    plt.figure(figsize=(6, 4))
    df['roc_auc'].plot(kind='bar', color=['#185FA5', '#E24B4A', '#BA7517'])
    plt.title('ROC-AUC por modelo')
    plt.ylabel('ROC-AUC')
    plt.ylim(0, 1)
    plt.tight_layout()
    png_out = OUT / 'roc_auc_comparison.png'
    plt.savefig(png_out, dpi=150)
    plt.close()
    print(f"PNG guardado: {png_out}")


if __name__ == '__main__':
    main()
