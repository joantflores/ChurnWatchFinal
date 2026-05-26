"""
ChurnWatch — Motor de Churn
===========================
Módulo responsable de:
  1. Construir la variable objetivo Churn a partir de señales de comportamiento.
  2. Aplicar feature engineering (encoding + normalización).
  3. Exponer funciones reutilizables para el notebook y la app Dash.

Uso básico:
    from churn_engine import construir_churn, preparar_features, resumen_churn

    df_raw   = pd.read_csv("shopping_trends.csv")
    df_churn = construir_churn(df_raw)          # agrega columna Churn
    X, y     = preparar_features(df_churn)      # listo para sklearn
    print(resumen_churn(df_churn))
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

BAJA_FRECUENCIA = ["Annually", "Every 3 Months", "Quarterly"]

PERCENTIL_UMBRAL = 0.25   

COLUMNAS_REQUERIDAS = [
    "Customer ID",
    "Frequency of Purchases",
    "Previous Purchases",
    "Subscription Status",
    "Purchase Amount (USD)",
    "Discount Applied",
    "Promo Code Used",
    "Payment Method",
]

COLUMNAS_OHE = [
    "Frequency of Purchases",
    "Subscription Status",
    "Discount Applied",
    "Promo Code Used",
    "Payment Method",
]

COLUMNAS_NUM = [
    "Previous Purchases",
    "Purchase Amount (USD)",
]

def construir_churn(
    df: pd.DataFrame,
    baja_frecuencia: list[str] = BAJA_FRECUENCIA,
    percentil: float = PERCENTIL_UMBRAL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Agrega la columna ``Churn`` al DataFrame.

    Regla de negocio (proxy variable):
        Churn = 1 si el cliente cumple dos condiciones:
            1. Frequency of Purchases  ∈  baja_frecuencia
            2. Previous Purchases      ≤  percentil 25 del dataset

    Parameters
    ----------
    df              : DataFrame con las columnas del dataset original.
    baja_frecuencia : Lista de valores de frecuencia que se consideran bajos.
    percentil       : Percentil (0-1) para el umbral de compras previas.
    verbose         : Si True imprime un resumen del resultado.

    Returns
    -------
    DataFrame con la columna ``Churn`` añadida (0 = activo, 1 = en riesgo).
    """

    # Verificar columnas mínimas
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Columnas faltantes en el DataFrame: {faltantes}\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    df = df.copy()

    umbral_compras = df["Previous Purchases"].quantile(percentil)

    mask = (
        df["Frequency of Purchases"].isin(baja_frecuencia)
        & (df["Previous Purchases"] <= umbral_compras)
    )

    df["Churn"] = mask.astype(int)

    if verbose:
        total       = len(df)
        n_churn     = df["Churn"].sum()
        churn_rate  = n_churn / total * 100
        umbral_int  = int(umbral_compras)

        print("=" * 50)
        print("  CONSTRUCCIÓN DE VARIABLE CHURN")
        print("=" * 50)
        print(f"  Frecuencias de baja actividad : {baja_frecuencia}")
        print(f"  Umbral compras previas (p{int(percentil*100)}) : <= {umbral_int}")
        print(f"  Clientes al inicio            : {total:,}")
        print(f"  Clientes con Churn = 1        : {n_churn:,}")
        print(f"  Churn Rate                    : {churn_rate:.2f}%")
        print("=" * 50)

        # Desglose por condición (útil para debuggear)
        cond_freq = df["Frequency of Purchases"].isin(baja_frecuencia).sum()
        cond_comp = (df["Previous Purchases"] <= umbral_compras).sum()
        print("\n  Clientes que cumplen cada condición (individualmente):")
        print(f"    Baja frecuencia             : {cond_freq:,}")
        print(f"    Pocas compras previas        : {cond_comp:,}")
        print(f"    Las 2 condiciones (Churn=1)  : {n_churn:,}\n")

    return df

def preparar_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Aplica One-Hot Encoding a variables categóricas y normaliza las numéricas.

    Parameters
    ----------
    df          : DataFrame con la columna ``Churn`` ya construida.
    scaler      : StandardScaler ya ajustado (None = crear uno nuevo).
    fit_scaler  : Si True ajusta el scaler con los datos (usar True solo
                  en entrenamiento, False en predicción con datos nuevos).

    Returns
    -------
    X       : DataFrame de features listo para sklearn.
    y       : Serie con la variable objetivo (Churn).
    scaler  : StandardScaler ajustado (guárdalo para predecir datos nuevos).
    """
    if "Churn" not in df.columns:
        raise ValueError(
            "El DataFrame no tiene columna 'Churn'. "
            "Llama primero a construir_churn()."
        )

    df_fe = df[COLUMNAS_OHE + COLUMNAS_NUM + ["Churn"]].copy()

    # One-Hot Encoding — drop_first evita multicolinealidad
    df_fe = pd.get_dummies(df_fe, columns=COLUMNAS_OHE, drop_first=True)

    y = df_fe.pop("Churn")
    X = df_fe.copy()

    # Normalización de columnas numéricas
    if scaler is None:
        scaler = StandardScaler()

    num_cols = [c for c in COLUMNAS_NUM if c in X.columns]
    if fit_scaler:
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])

    return X, y, scaler

def resumen_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con el churn rate desglosado por variable categórica.
    Útil para exploración y para alimentar gráficas en Dash.
    """
    if "Churn" not in df.columns:
        raise ValueError("El DataFrame no tiene columna 'Churn'.")

    resultados = {}
    for col in ["Frequency of Purchases", "Subscription Status",
                "Discount Applied", "Promo Code Used"]:
        if col in df.columns:
            tasa = (
                df.groupby(col)["Churn"]
                .mean()
                .mul(100)
                .round(2)
                .rename("Churn Rate (%)")
                .reset_index()
            )
            tasa["Variable"] = col
            resultados[col] = tasa

    return pd.concat(resultados.values(), ignore_index=True)

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shopping_trends.csv"
    print(f"\nCargando: {csv_path}\n")

    df_raw = pd.read_csv(csv_path)

    print(">>> DEFINICIÓN ORIGINAL (solo 'Annually'):")
    df_orig = construir_churn(df_raw, baja_frecuencia=["Annually"], verbose=True)

    print("\n>>> DEFINICIÓN CORREGIDA (Annually + Every 3 Months + Quarterly):")
    df_corr = construir_churn(df_raw, verbose=True)

    print("\n>>> FEATURE ENGINEERING:")
    X, y, scaler = preparar_features(df_corr)
    print(f"  Shape de X : {X.shape}")
    print(f"  Columnas   : {X.columns.tolist()}")
    print(f"  Balance    : Churn=0 → {(y==0).sum():,}  |  Churn=1 → {(y==1).sum():,}")

    print("\n>>> CHURN RATE POR VARIABLE:")
    print(resumen_churn(df_corr).to_string(index=False))
