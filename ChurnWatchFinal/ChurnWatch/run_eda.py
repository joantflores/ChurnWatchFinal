"""
ChurnWatch - EDA multi-dataset for Avance 2.

Genera analisis exploratorio para:
1) Customer Shopping Trends
2) Online Retail II (si el archivo existe)
3) Telco Customer Churn

Salida:
- outputs/eda/<dataset>/summary.txt
- outputs/eda/<dataset>/hist_*.png
- outputs/eda/<dataset>/box_*.png
- outputs/eda/<dataset>/top_categories_*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "eda"

DEFAULT_DATASETS = {
    "shopping_trends": ROOT / "shopping_trends.csv",
    "online_retail_ii": ROOT / "OnlineRetailII.csv",
    "telco_churn": ROOT / "TelcoCustomerChurn.csv",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding_errors="ignore")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Formato no soportado: {path.name}")


def safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def safe_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def write_summary(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    nulls = df.isnull().sum().sort_values(ascending=False)
    dups = int(df.duplicated().sum())

    lines: list[str] = []
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Shape: {df.shape[0]} filas x {df.shape[1]} columnas")
    lines.append("")
    lines.append("Columnas:")
    lines.extend([f"- {c}" for c in df.columns])
    lines.append("")
    lines.append("Valores nulos por columna (top 20):")
    lines.extend([f"- {idx}: {val}" for idx, val in nulls.head(20).items()])
    lines.append("")
    lines.append(f"Duplicados totales: {dups}")
    lines.append("")
    lines.append("Estadistica descriptiva (numericas):")
    if safe_numeric_columns(df):
        lines.append(df.describe().to_string())
    else:
        lines.append("No hay columnas numericas.")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_numeric_distributions(df: pd.DataFrame, out_dir: Path, focus_cols: Iterable[str] | None = None) -> None:
    numeric_cols = safe_numeric_columns(df)
    if not numeric_cols:
        return

    focus = [c for c in (focus_cols or []) if c in df.columns]
    target_cols = focus if focus else numeric_cols[:8]

    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        plt.figure(figsize=(8, 4))
        sns.histplot(series, kde=True, bins=30, color="#185FA5")
        plt.title(f"Distribucion - {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8, 2.4))
        sns.boxplot(x=series, color="#5DCAA5")
        plt.title(f"Boxplot - {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"box_{col}.png", dpi=140)
        plt.close()


def plot_top_categories(df: pd.DataFrame, out_dir: Path) -> None:
    cat_cols = safe_categorical_columns(df)[:6]
    for col in cat_cols:
        top = df[col].astype(str).value_counts(dropna=False).head(10)
        if top.empty:
            continue
        plt.figure(figsize=(8, 4))
        sns.barplot(x=top.values, y=top.index, orient="h", color="#185FA5")
        plt.title(f"Top categorias - {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"top_categories_{col}.png", dpi=140)
        plt.close()


def telco_churn_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    churn_col = None
    for col in df.columns:
        if col.lower().strip() == "churn":
            churn_col = col
            break
    if churn_col is None:
        return

    counts = df[churn_col].astype(str).value_counts()
    plt.figure(figsize=(5, 3.5))
    sns.barplot(x=counts.index, y=counts.values, color="#E24B4A")
    plt.title("Distribucion de Churn (Telco)")
    plt.ylabel("Clientes")
    plt.tight_layout()
    plt.savefig(out_dir / "churn_distribution.png", dpi=140)
    plt.close()


def run() -> None:
    ensure_dir(OUTPUT_ROOT)
    print("=== EDA ChurnWatch ===")
    for name, path in DEFAULT_DATASETS.items():
        if not path.exists():
            print(f"[SKIP] {name}: no se encontro {path.name}")
            continue

        print(f"[OK] Procesando {name}: {path.name}")
        out_dir = OUTPUT_ROOT / name
        ensure_dir(out_dir)
        df = read_dataset(path)

        write_summary(df, name, out_dir)
        if name == "shopping_trends":
            plot_numeric_distributions(
                df,
                out_dir,
                focus_cols=["Previous Purchases", "Purchase Amount (USD)", "Age", "Review Rating"],
            )
        else:
            plot_numeric_distributions(df, out_dir)
        plot_top_categories(df, out_dir)

        if name == "telco_churn":
            telco_churn_distribution(df, out_dir)

    print(f"EDA completado. Revisa: {OUTPUT_ROOT}")


if __name__ == "__main__":
    run()
