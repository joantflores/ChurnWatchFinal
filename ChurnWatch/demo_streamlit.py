from pathlib import Path

import pandas as pd
import streamlit as st

from churn_engine import (
    CHURN_UMBRAL_ENTRENAMIENTO,
    FREQUENCIAS_COMPRA,
    calcular_churn_regla,
    explicar_factores_churn,
    normalizar_columnas_df,
    probabilidad_desde_factores,
)

st.title("ChurnWatch")
st.write(
    "Probabilidad con **5 factores**: frecuencia, compras previas, "
    "suscripción, descuento y monto (USD)."
)

with st.form("input_form"):
    st.subheader("Datos del cliente")
    frequency = st.selectbox("Frecuencia de compras", FREQUENCIAS_COMPRA, index=3)
    previous = st.number_input("Compras previas", min_value=0, max_value=1000, value=5)
    amount = st.number_input("Monto total (USD)", min_value=0.0, value=50.0, format="%.2f")
    subscription = st.selectbox("Suscripción activa", ["Sí", "No"])
    discount = st.selectbox("Descuento aplicado", ["Sí", "No"])
    submitted = st.form_submit_button("Predecir")

if submitted:
    subs = "Yes" if subscription.lower().startswith("s") else "No"
    disc = "Yes" if discount.lower().startswith("s") else "No"

    df = pd.DataFrame(
        [
            {
                "frequency_of_purchases": frequency,
                "previous_purchases": int(previous),
                "purchase_amount_usd": float(amount),
                "subscription_status": subs,
                "discount_applied": disc,
            }
        ]
    )
    df = normalizar_columnas_df(df)
    churn_s, umbral = calcular_churn_regla(df, umbral_fijo=CHURN_UMBRAL_ENTRENAMIENTO)
    pct = probabilidad_desde_factores(
        frequency, previous, subscription=subs, discount=disc, amount=amount, umbral=umbral
    )
    factores = explicar_factores_churn(
        frequency, previous, subscription=subs, discount=disc, amount=amount, umbral=umbral
    )

    st.metric("Churn (regla)", "Sí" if int(churn_s.iloc[0]) else "No")
    st.metric("Probabilidad", f"{pct}%")
    st.caption(factores)

    if pct >= 65:
        st.warning("Riesgo: Alto")
    elif pct >= 40:
        st.info("Riesgo: Medio")
    else:
        st.success("Riesgo: Bajo")
