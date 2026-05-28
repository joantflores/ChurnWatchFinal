"""
ChurnWatch — Motor de Churn
===========================
Regla de negocio, predicción ML y probabilidades alineadas para app y scripts.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Regla de negocio (documento PF01)
BAJA_FRECUENCIA = ["Annually", "Every 3 Months", "Quarterly"]
BAJA_FRECUENCIA_CANON = {"annually", "every 3 months", "quarterly"}
CHURN_UMBRAL_ENTRENAMIENTO = 13  # percentil 25 en Customer Shopping Trends
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

COL_ALIASES = {
    "customer_id": ["customer_id", "id", "customer", "cliente", "customer id"],
    "frequency_of_purchases": [
        "frequency_of_purchases",
        "frequency",
        "frecuencia",
        "frequency of purchases",
    ],
    "previous_purchases": [
        "previous_purchases",
        "purchases",
        "compras",
        "previous purchases",
    ],
    "subscription_status": [
        "subscription_status",
        "subscription",
        "suscripcion",
        "subscription status",
    ],
    "purchase_amount_usd": [
        "purchase_amount_usd",
        "amount",
        "monto",
        "purchase amount (usd)",
        "purchase amount",
    ],
    "discount_applied": [
        "discount_applied",
        "discount",
        "descuento",
        "discount applied",
    ],
    "promo_code_used": ["promo_code_used", "promo code used", "promo code"],
    "payment_method": ["payment_method", "payment method", "payment"],
}

# Valores del dataset Customer Shopping Trends (ingreso manual y referencia)
FREQUENCIAS_COMPRA = [
    "Weekly",
    "Bi-Weekly",
    "Fortnightly",
    "Monthly",
    "Quarterly",
    "Every 3 Months",
    "Annually",
]

FREQ_ALIASES = {
    "weekly": "weekly",
    "bi-weekly": "bi-weekly",
    "biweekly": "bi-weekly",
    "fortnightly": "fortnightly",
    "monthly": "monthly",
    "quarterly": "quarterly",
    "every 3 months": "every 3 months",
    "annually": "annually",
    "semanal": "weekly",
    "quincenal": "fortnightly",
    "mensual": "monthly",
    "trimestral": "quarterly",
    "cada 3 meses": "every 3 months",
    "anual": "annually",
}


def canonical_name(column: str) -> str:
    return (
        column.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def normalizar_columnas_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nombres estándar snake_case (compatible con CSV original y derivados)."""
    out = df.copy()
    out.columns = [canonical_name(c) for c in out.columns]
    rename: dict[str, str] = {}
    lower_cols = set(out.columns)
    for standard, aliases in COL_ALIASES.items():
        if standard in lower_cols:
            continue
        for alias in aliases:
            if alias in lower_cols:
                rename[alias] = standard
                break
    return out.rename(columns=rename)


def normalizar_frecuencia(value: Any) -> str:
    key = str(value).lower().strip()
    return FREQ_ALIASES.get(key, key)


def es_baja_frecuencia(value: Any) -> bool:
    return normalizar_frecuencia(value) in BAJA_FRECUENCIA_CANON


def umbral_compras_previas(
    previous: pd.Series,
    *,
    umbral_fijo: int | None = CHURN_UMBRAL_ENTRENAMIENTO,
    percentil: float = PERCENTIL_UMBRAL,
) -> int:
    prev = pd.to_numeric(previous, errors="coerce").fillna(0)
    if umbral_fijo is not None:
        return int(umbral_fijo)
    return int(prev.quantile(percentil))


def calcular_churn_regla(
    df: pd.DataFrame,
    *,
    umbral_fijo: int | None = CHURN_UMBRAL_ENTRENAMIENTO,
    percentil: float = PERCENTIL_UMBRAL,
) -> tuple[pd.Series, int]:
    """
    Churn proxy: 1 si baja frecuencia Y compras previas <= umbral.
    """
    work = normalizar_columnas_df(df)
    if "frequency_of_purchases" not in work.columns or "previous_purchases" not in work.columns:
        raise ValueError(
            "Se requieren columnas frequency_of_purchases y previous_purchases. "
            f"Disponibles: {work.columns.tolist()}"
        )

    prev = pd.to_numeric(work["previous_purchases"], errors="coerce").fillna(0)
    umbral = umbral_compras_previas(prev, umbral_fijo=umbral_fijo, percentil=percentil)
    baja_freq = work["frequency_of_purchases"].map(es_baja_frecuencia)
    churn = (baja_freq & (prev <= umbral)).astype(int)
    return churn, umbral


# Columnas que alimentan la probabilidad en la app
FACTORES_PROBABILIDAD = [
    "frequency_of_purchases",
    "previous_purchases",
    "subscription_status",
    "discount_applied",
    "purchase_amount_usd",
]

MONTO_REFERENCIA_USD = 60.0  # media aproximada del dataset


def _valor_es_no(value: Any) -> bool:
    return str(value).lower().strip() in ("no", "false", "0", "n")


def probabilidad_desde_factores(
    frequency: Any,
    previous: Any,
    *,
    subscription: Any = "Yes",
    discount: Any = "Yes",
    amount: Any = MONTO_REFERENCIA_USD,
    umbral: int = CHURN_UMBRAL_ENTRENAMIENTO,
) -> int:
    """
    Probabilidad de churn (0–99) con los 5 factores del negocio:
      1. Frecuencia de compra
      2. Cantidad de compras previas
      3. Suscripción activa
      4. Descuento aplicado
      5. Monto total (USD)
    """
    prev = int(pd.to_numeric(previous, errors="coerce") or 0)
    monto = float(pd.to_numeric(amount, errors="coerce") or MONTO_REFERENCIA_USD)
    freq = normalizar_frecuencia(frequency)

    riesgo_frecuencia = {
        "weekly": 0,
        "bi-weekly": 4,
        "fortnightly": 5,
        "monthly": 10,
        "quarterly": 20,
        "every 3 months": 24,
        "annually": 30,
    }
    score = riesgo_frecuencia.get(freq, 12)

    if prev <= umbral:
        score += 20 + max(0, umbral - prev) * 2
    elif prev <= 25:
        score += 7

    if _valor_es_no(subscription):
        score += 12

    if _valor_es_no(discount):
        score += 6

    if monto < 35:
        score += 8
    elif monto < 50:
        score += 3
    elif monto > 80:
        score -= 5

    # Coherente con la etiqueta churn (frecuencia baja + pocas compras)
    if es_baja_frecuencia(frequency) and prev <= umbral:
        score = max(score, 75 + max(0, umbral - prev) * 2)

    return int(np.clip(score, 0, 99))


def probabilidad_desde_regla(
    frequency: Any,
    previous: Any,
    *,
    umbral: int = CHURN_UMBRAL_ENTRENAMIENTO,
    **kwargs: Any,
) -> int:
    """Compatibilidad: delega en probabilidad_desde_factores."""
    return probabilidad_desde_factores(
        frequency,
        previous,
        subscription=kwargs.get("subscription", "Yes"),
        discount=kwargs.get("discount", "Yes"),
        amount=kwargs.get("amount", MONTO_REFERENCIA_USD),
        umbral=umbral,
    )


def explicar_factores_churn(
    frequency: Any,
    previous: Any,
    *,
    subscription: Any = "Yes",
    discount: Any = "Yes",
    amount: Any = MONTO_REFERENCIA_USD,
    umbral: int = CHURN_UMBRAL_ENTRENAMIENTO,
) -> str:
    """Resumen de los 5 factores usados en la probabilidad."""
    prev = int(pd.to_numeric(previous, errors="coerce") or 0)
    monto = float(pd.to_numeric(amount, errors="coerce") or MONTO_REFERENCIA_USD)
    subs = "No" if _valor_es_no(subscription) else "Sí"
    disc = "No" if _valor_es_no(discount) else "Sí"
    baja = es_baja_frecuencia(frequency)

    return (
        f"Frecuencia: {frequency} ({'baja' if baja else 'activa'}) · "
        f"Compras previas: {prev} · Suscripción: {subs} · "
        f"Descuento: {disc} · Monto: ${monto:.0f} USD"
    )


def rellenar_factores_probabilidad(df: pd.DataFrame) -> pd.DataFrame:
    """Valores por defecto si el CSV no trae columnas opcionales."""
    out = normalizar_columnas_df(df).copy()
    if "subscription_status" not in out.columns:
        out["subscription_status"] = "Yes"
    if "discount_applied" not in out.columns:
        out["discount_applied"] = "Yes"
    if "purchase_amount_usd" not in out.columns:
        out["purchase_amount_usd"] = MONTO_REFERENCIA_USD
    return out


def datos_suficientes_para_ml(
    df: pd.DataFrame,
    payload: dict[str, Any],
    *,
    min_frac_categoricas: float = 0.75,
) -> bool:
    """ML fiable solo si el archivo trae la mayoría de columnas categóricas del entrenamiento."""
    cat_cols = payload.get("categorical_features") or []
    if not cat_cols:
        return False

    work = normalizar_columnas_df(df)
    presentes = 0
    for col in cat_cols:
        if col not in work.columns:
            continue
        serie = work[col].astype(str).str.strip()
        if serie.replace("", np.nan).replace("nan", np.nan).notna().any():
            if (serie.str.lower() != "missing").any():
                presentes += 1

    return presentes >= max(1, int(len(cat_cols) * min_frac_categoricas))


def preparar_dataframe_modelo(df: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    """DataFrame con columnas que espera el pipeline sklearn guardado en best_model.pkl."""
    work = normalizar_columnas_df(df)
    input_cols = payload.get("input_columns") or []
    num_cols = payload.get("numeric_features") or []
    cat_cols = payload.get("categorical_features") or []

    model_df = pd.DataFrame(index=work.index)
    for col in input_cols:
        if col in work.columns:
            model_df[col] = work[col]
        elif col in num_cols:
            model_df[col] = 0
        else:
            model_df[col] = "missing"

    for col in num_cols:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

    for col in cat_cols:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna("missing").astype(str)

    return model_df


def predecir_probabilidades_ml(
    df: pd.DataFrame,
    payload: dict[str, Any],
) -> np.ndarray:
    model = payload.get("modelo")
    if model is None:
        raise ValueError("Payload sin modelo.")
    model_df = preparar_dataframe_modelo(df, payload)
    return model.predict_proba(model_df)[:, 1]


def construir_churn(
    df: pd.DataFrame,
    baja_frecuencia: list[str] | None = None,
    percentil: float = PERCENTIL_UMBRAL,
    umbral_fijo: int | None = CHURN_UMBRAL_ENTRENAMIENTO,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Agrega columna ``Churn`` (compatible con CSV original en Title Case).
    """
    df = df.copy()
    churn_s, umbral = calcular_churn_regla(
        df, umbral_fijo=umbral_fijo, percentil=percentil
    )
    df["Churn"] = churn_s.values

    if verbose:
        total = len(df)
        n_churn = int(churn_s.sum())
        churn_rate = n_churn / total * 100 if total else 0.0
        print("=" * 50)
        print("  CONSTRUCCIÓN DE VARIABLE CHURN")
        print("=" * 50)
        print(f"  Frecuencias baja actividad : {sorted(BAJA_FRECUENCIA_CANON)}")
        print(f"  Umbral compras previas      : <= {umbral}")
        print(f"  Clientes                    : {total:,}")
        print(f"  Churn = 1                   : {n_churn:,}")
        print(f"  Churn rate                  : {churn_rate:.2f}%")
        print("=" * 50)

    return df


def preparar_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Feature engineering reducido (notebooks); el .pkl usa pipeline completo."""
    if "Churn" not in df.columns:
        raise ValueError("Llama primero a construir_churn().")

    title_map = {
        "frequency_of_purchases": "Frequency of Purchases",
        "previous_purchases": "Previous Purchases",
        "subscription_status": "Subscription Status",
        "purchase_amount_usd": "Purchase Amount (USD)",
        "discount_applied": "Discount Applied",
        "promo_code_used": "Promo Code Used",
        "payment_method": "Payment Method",
    }
    work = normalizar_columnas_df(df)
    for snake, title in title_map.items():
        if snake in work.columns and title not in work.columns:
            work[title] = work[snake]

    df_fe = work[
        [c for c in COLUMNAS_OHE + COLUMNAS_NUM + ["Churn"] if c in work.columns]
    ].copy()
    if "Churn" not in df_fe.columns and "churn" in work.columns:
        df_fe["Churn"] = work["churn"]

    ohe_cols = [c for c in COLUMNAS_OHE if c in df_fe.columns]
    df_fe = pd.get_dummies(df_fe, columns=ohe_cols, drop_first=True)

    y = df_fe.pop("Churn")
    X = df_fe.copy()

    if scaler is None:
        scaler = StandardScaler()

    num_cols = [c for c in COLUMNAS_NUM if c in X.columns]
    if num_cols:
        if fit_scaler:
            X[num_cols] = scaler.fit_transform(X[num_cols])
        else:
            X[num_cols] = scaler.transform(X[num_cols])

    return X, y, scaler


def resumen_churn(df: pd.DataFrame) -> pd.DataFrame:
    if "Churn" not in df.columns:
        if "churn" in df.columns:
            df = df.copy()
            df["Churn"] = df["churn"]
        else:
            raise ValueError("El DataFrame no tiene columna 'Churn'.")

    resultados = {}
    freq_col = "Frequency of Purchases" if "Frequency of Purchases" in df.columns else "frequency_of_purchases"
    for col in [freq_col, "Subscription Status", "subscription_status",
                "Discount Applied", "discount_applied", "Promo Code Used", "promo_code_used"]:
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
    df_corr = construir_churn(df_raw, verbose=True)
    X, y, _ = preparar_features(df_corr)
    print(f"  Shape de X : {X.shape}")
    print(f"  Balance    : Churn=0 → {(y == 0).sum():,}  |  Churn=1 → {(y == 1).sum():,}")
