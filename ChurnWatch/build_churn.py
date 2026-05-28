"""Construye la variable proxy `churn` para Customer Shopping Trends.

Regla usada en el documento del proyecto:
churn = 1 si el cliente tiene baja frecuencia de compra
("Annually", "Every 3 Months" o "Quarterly") y pocas compras previas
(Previous Purchases <= percentil 25, que en este dataset es 13).
"""

from pathlib import Path

import pandas as pd

from churn_engine import (
    CHURN_UMBRAL_ENTRENAMIENTO,
    calcular_churn_regla,
    normalizar_columnas_df,
)

ROOT = Path(__file__).resolve().parent
INFILE = ROOT / "shopping_trends.csv"
OUTFILE = ROOT / "shopping_trends_churn.csv"


def build_churn(df: pd.DataFrame, p25: int | None = CHURN_UMBRAL_ENTRENAMIENTO) -> pd.DataFrame:
    df = normalizar_columnas_df(df)
    churn, _ = calcular_churn_regla(df, umbral_fijo=p25)
    df["churn"] = churn.values
    return df


def main() -> None:
    if not INFILE.exists():
        print(f"No se encontro {INFILE}. Coloca el CSV de Customer Shopping Trends como 'shopping_trends.csv'")
        return

    # intentar leer con utf-8, si falla probar latin-1 y finalmente un decode con replace
    try:
        df = pd.read_csv(INFILE, encoding="utf-8")

    except UnicodeDecodeError:
        try:
            df = pd.read_csv(INFILE, encoding="latin-1")

        except Exception:
            # ultimo recurso: leer en binario y decodificar reemplazando bytes invalidos
            with open(INFILE, "rb") as f:
                raw = f.read()

            import io as _io
            text = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(_io.StringIO(text))

    out = build_churn(df)
    out.to_csv(OUTFILE, index=False)
    counts = out["churn"].value_counts()
    churn_rate = out["churn"].mean() * 100
    print(f"Guardado: {OUTFILE}")
    print(f"Balance de clases:\n{counts.to_string()}")
    print(f"Churn rate proxy: {churn_rate:.2f}%")

if __name__ == "__main__":
    main()
