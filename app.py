import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

# Load models
churn_model = joblib.load('churn_model.joblib')
revenue_model = joblib.load('revenue_forecast_model.joblib')

# Page config
st.set_page_config(
    page_title="AI‑Powered Churn & Revenue Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔍 AI‑Powered Customer Insights Dashboard")
st.markdown("Predict customer churn and forecast revenue - interactive, explainable, and batch‑capable.")

# Sidebar inputs with tooltips
st.sidebar.header("📥 Customer Input")
age = st.sidebar.slider(
    "Age",
    min_value=18, max_value=80, value=35,
    help="Customer’s current age in years"
)
tenure_months = st.sidebar.slider(
    "Tenure (Months)",
    min_value=0, max_value=120, value=12,
    help="How many months the customer has been active"
)
monthly_spend = st.sidebar.slider(
    "Monthly Spend ($)",
    min_value=10, max_value=500, value=70,
    help="Average monthly amount spent by the customer"
)

# Prepare input and run predictions
input_arr = np.array([[age, tenure_months, monthly_spend]])
churn_prob = churn_model.predict_proba(input_arr)[0][1]
churn_pred = churn_model.predict(input_arr)[0]
revenue_pred = revenue_model.predict(np.array([[tenure_months, churn_pred]]))[0]

# KPI Cards
st.markdown("### 📊 Prediction Results")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Churn Risk", f"{churn_prob*100:.1f}%", delta_color="inverse")
with col2:
    st.metric("Churn Prediction", "Yes" if churn_pred == 1 else "No")
with col3:
    st.metric("Expected Monthly Revenue", f"${revenue_pred:.2f}")
style_metric_cards(
    background_color="linear-gradient(135deg, #667eea, #764ba2)",  # Purple/blue gradient
    border_color="#CCCCCC",
    border_left_color="#764ba2"
)

st.markdown("---")

# Gauge Chart for Churn Risk
st.markdown("### 🚦 Churn Risk Gauge")
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_prob*100,
    domain={'x': [0,1], 'y': [0,1]},
    title={'text': "Churn Risk (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "red" if churn_prob > 0.5 else "green"},
        'steps': [
            {'range': [0, 50],  'color': "#90ee90"},
            {'range': [50, 75], 'color': "#ffcccb"},
            {'range': [75, 100],'color': "#ff6666"}
        ]
    }
))
st.plotly_chart(gauge, use_container_width=True)

# Revenue Forecast over Tenure
st.markdown("### 📈 Revenue Forecast over Tenure")
months = np.arange(1, 121)
sim_rev = revenue_model.predict(np.column_stack([months, [churn_pred]*len(months)]))
rev_chart = go.Figure()
rev_chart.add_trace(go.Scatter(
    x=months, y=sim_rev,
    mode='lines+markers',
    name='Forecasted Revenue'
))
rev_chart.update_layout(
    title="Revenue vs. Tenure",
    xaxis_title="Tenure (Months)",
    yaxis_title="Revenue ($)",
    template="plotly_white"
)
st.plotly_chart(rev_chart, use_container_width=True)

st.markdown("---")

# Download single prediction report
st.markdown("### 📥 Download Single‑Customer Report")
report_df = pd.DataFrame({
    "Age": [age],
    "Tenure_Months": [tenure_months],
    "Monthly_Spend": [monthly_spend],
    "Churn_Risk_%": [round(churn_prob*100, 2)],
    "Churn_Prediction": ["Yes" if churn_pred == 1 else "No"],
    "Revenue_Forecast_$": [round(revenue_pred, 2)]
})
csv = report_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='single_customer_report.csv',
    mime='text/csv'
)

st.markdown("---")

# Feature importance (mocked example)
st.markdown("### 🧠 Feature Importance")
feat_imp = pd.DataFrame({
    "Feature": ["Tenure_Months", "Monthly_Spend", "Age"],
    "Importance": [0.5, 0.3, 0.2]
})
fig_imp = go.Figure(go.Bar(
    x=feat_imp["Importance"],
    y=feat_imp["Feature"],
    orientation='h',
    marker_color="teal"
))
fig_imp.update_layout(
    title="Estimated Feature Importance",
    xaxis_title="Importance Score",
    yaxis_title="Feature",
    template="plotly_white"
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# Batch predictions from uploaded CSV
st.markdown("### 📂 Batch Prediction")
st.caption("Upload a CSV with columns: age, tenure_months, monthly_spend")
uploaded = st.file_uploader("", type=['csv'])
if uploaded:
    df_batch = pd.read_csv(uploaded)
    if set(['age','tenure_months','monthly_spend']).issubset(df_batch.columns):
        X_batch = df_batch[['age','tenure_months','monthly_spend']].values
        df_batch["Churn_Risk_%"]   = np.round(churn_model.predict_proba(X_batch)[:,1]*100,2)
        df_batch["Churn_Prediction"] = churn_model.predict(X_batch).astype(int)
        df_batch["Churn_Prediction"] = df_batch["Churn_Prediction"].map({1:"Yes", 0:"No"})
        rev_inputs = np.column_stack([df_batch['tenure_months'], churn_model.predict(X_batch)])
        df_batch["Revenue_Forecast_$"] = np.round(revenue_model.predict(rev_inputs), 2)

        st.dataframe(df_batch, use_container_width=True)
        batch_csv = df_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Batch Results",
            data=batch_csv,
            file_name='batch_predictions.csv',
            mime='text/csv'
        )
    else:
        st.error("CSV must contain columns: age, tenure_months, monthly_spend")

st.markdown("---")
st.caption("Built by Pranjali • Powered by AI • Demo only")
