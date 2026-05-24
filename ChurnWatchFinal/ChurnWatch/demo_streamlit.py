from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / 'best_model.pkl'
st.title('ChurnWatch')
st.write('Ingresa los datos y presiona Predecir.')

model_payload = None

if MODEL_FILE.exists():
    try:
        model_payload = joblib.load(MODEL_FILE)

    except Exception:
        model_payload = None

with st.form('input_form'):
    st.subheader('Customer data')
    frequency = st.selectbox('Frequencia de compras', ['Semanal','Mensual','Quincenal','Cada 3 meses','Anual'])
    previous = st.number_input('Compras previas', min_value=0, max_value=1000, value=5)
    amount = st.number_input('Monto total (USD)', min_value=0.0, value=50.0, format='%.2f')
    subscription = st.selectbox('¿Suscripcion activa?', ['Si','No'])
    discount = st.selectbox('¿Descuento aplicado?', ['Si','No'])
    submitted = st.form_submit_button('Predecir')

    input_dict = {
        'frequency_of_purchases': frequency,
        'previous_purchases': int(previous),
        'purchase_amount_usd': float(amount),
        'subscription_status': subscription,
        'discount_applied': discount,
    }

if submitted:
    st.subheader('Input summary')
    st.table(pd.DataFrame([input_dict]))
    df = pd.DataFrame([input_dict])
    df.columns = [c.lower().strip() for c in df.columns]

    def heuristic_score(freq, prev, subs, disc):
        base = {
            'semanal':0,'quincenal':5,'fortnightly':5,'mensual':15,'trimestral':55,'cada 3 meses':70,'anual':85
        }

        score = base.get(str(freq).lower(), 30)

        if prev <= 13:
            score += 20

        elif prev <= 25:
            score += 8

        if str(subs).lower() in ('no','false','0'):
            score += 10

        if str(disc).lower() in ('no','false','0'):
            score += 5

        return int(np.clip(score, 0, 99))

    pct = None

    if model_payload is not None and isinstance(model_payload, dict):

        model = model_payload.get('modelo')

        try:
            from churn_engine import preparar_features

            model_df = pd.DataFrame({
                'Frequency of Purchases': df.get('frequency_of_purchases',''),
                'Subscription Status': df.get('subscription_status',''),
                'Discount Applied': df.get('discount_applied',''),
                'Promo Code Used': 'No',
                'Payment Method': 'Unknown',
                'Previous Purchases': pd.to_numeric(df.get('previous_purchases',0), errors='coerce').fillna(0),
                'Purchase Amount (USD)': pd.to_numeric(df.get('purchase_amount_usd',0), errors='coerce').fillna(0),
                'Churn': 0,
            })

            X_pred, _, _ = preparar_features(model_df, scaler=model_payload.get('scaler'), fit_scaler=False)

            if model_payload.get('feature_columns'):

                X_pred = X_pred.reindex(columns=model_payload.get('feature_columns'), fill_value=0)

            prob = float(model.predict_proba(X_pred)[:,1][0])
            pct = int(round(prob*100))
            st.success('Prediction completed')
            st.metric('Churn probability', '%d%%' % pct)

        except Exception:
            pct = heuristic_score(frequency, previous, subscription, discount)
            st.info('Modelo ML no disponible; usar heuristica')
            st.metric('Probabilidad de churn', '%d%%' % pct)

    else:
        pct = heuristic_score(frequency, previous, subscription, discount)
        st.metric('Probabilidad de churn', '%d%%' % pct)

    if pct is not None:
        if pct >= 65:
            st.warning('Riesgo: Alto')
            st.error('Recomendacion: contactar con oferta personalizada')

        elif pct >= 40:
            st.info('Riesgo: Medio')
            st.warning('Recomendacion: enviar recordatorio o incentivo')

        else:
            st.success('Riesgo: Bajo')
            st.info('Recomendacion: mantener contacto regular')