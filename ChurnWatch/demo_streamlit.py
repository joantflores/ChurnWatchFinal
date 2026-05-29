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

# Título principal de la app
st.title("ChurnWatch")

# Descripción breve del modelo de probabilidad
st.write(
    "Probabilidad con **5 factores**: frecuencia, compras previas, "
    "suscripción, descuento y monto (USD)."
)

# Formulario principal de entrada de datos
with st.form("input_form"):

    # Encabezado de la sección de captura
    st.subheader("Datos del cliente")

    # Selección de frecuencia de compra
    frequency = st.selectbox(
        "Frecuencia de compras",
        FREQUENCIAS_COMPRA,
        index=3,
    )

    # Número de compras anteriores del cliente
    previous = st.number_input(
        "Compras previas",
        min_value=0,
        max_value=1000,
        value=5,
    )

    # Monto total promedio gastado por el cliente
    amount = st.number_input(
        "Monto total (USD)",
        min_value=0.0,
        value=50.0,
        format="%.2f",
    )

    # Estado de suscripción del cliente
    subscription = st.selectbox(
        "Suscripción activa",
        ["Sí", "No"],
    )

    # Indica si el cliente utilizó descuento
    discount = st.selectbox(
        "Descuento aplicado",
        ["Sí", "No"],
    )

    # Botón para ejecutar la predicción
    submitted = st.form_submit_button("Predecir")

# Ejecutar predicción solo si el usuario envió el formulario
if submitted:

    # Convertir valores Sí/No a formato esperado por el motor
    subs = "Yes" if subscription.lower().startswith("s") else "No"
    disc = "Yes" if discount.lower().startswith("s") else "No"

    # Construir DataFrame con los datos ingresados
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

    # Normalizar nombres de columnas
    df = normalizar_columnas_df(df)

    # Calcular churn usando la regla de negocio
    churn_s, umbral = calcular_churn_regla(
        df,
        umbral_fijo=CHURN_UMBRAL_ENTRENAMIENTO,
    )

    # Calcular probabilidad estimada de abandono
    pct = probabilidad_desde_factores(
        frequency,
        previous,
        subscription=subs,
        discount=disc,
        amount=amount,
        umbral=umbral,
    )

    # Generar explicación de factores usados
    factores = explicar_factores_churn(
        frequency,
        previous,
        subscription=subs,
        discount=disc,
        amount=amount,
        umbral=umbral,
    )

    # Mostrar resultado binario de churn
    st.metric(
        "Churn (regla)",
        "Sí" if int(churn_s.iloc[0]) else "No",
    )

    # Mostrar porcentaje de probabilidad
    st.metric(
        "Probabilidad",
        f"{pct}%",
    )

    # Mostrar explicación resumida de factores
    st.caption(factores)

    # Mostrar nivel de riesgo según porcentaje
    if pct >= 65:
        st.warning("Riesgo: Alto")

    elif pct >= 40:
        st.info("Riesgo: Medio")

    else:
        st.success("Riesgo: Bajo")