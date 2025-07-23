import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go

# ========== Data & Model Loaders ==========

def load_data():
    df = pd.read_csv("data/churn_data.csv")
    df["churn"] = df["subscription_status"].replace({'Active':0,'Cancelled':1})
    return df

def load_xgb_model():
    with open('model/model_2.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# ========== Preprocessing ==========

def preprocess(user_input):
    df = pd.DataFrame([user_input])
    df['auto_renewal'] = df['auto_renewal'].map({'Yes':1,'No':0})
    df['roaming_last_3m'] = df['roaming_last_3m'].map({'Yes':1,'No':0})
    df['loyalty_member'] = df['loyalty_member'].map({'Yes':1,'No':0})
    df['on_time_payments'] = df['on_time_payments'].map({'Yes':1,'No':0})
    df['discount_last_renewal'] = df['discount_last_renewal'].map({'Yes':1,'No':0})
    df['downgraded_last_6m'] = df['downgraded_last_6m'].map({'Yes':1,'No':0})
    df['last_offer_accepted'] = df['last_offer_accepted'].map({'Yes':1,'No':0})

    df = pd.get_dummies(df, columns = ["gender"], prefix = 'gender')
    df = pd.get_dummies(df, columns = ['region'], prefix = 'region')
    df = pd.get_dummies(df, columns = ['customer_type'], prefix = 'customer_type')
    df = pd.get_dummies(df, columns = ['plan_type'], prefix = 'plan_type')
    df = pd.get_dummies(df, columns = ['plan_tier'], prefix = 'plan_tier')
    df = pd.get_dummies(df, columns = ['payment_method'], prefix = 'payment_method')
    


    MODEL_FEATURES = ['age', 'monthly_fee_€', 'contract_length_months', 'auto_renewal',
       'avg_monthly_data_gb', 'avg_monthly_voice_minutes',
       'sms_per_month', 'roaming_last_3m', 'loyalty_member',
       'app_logins_per_month', 'dropped_calls_last_month',
       'avg_download_speed_mbps', 'coverage_complaints_6m', 'csat_score',
       'on_time_payments', 'late_payments_6m', 'outstanding_balance_€',
       'discount_last_renewal', 'downgraded_last_6m',
       'last_offer_accepted', 'campaign_ctr', 'support_tickets_6m',
       'avg_resolution_time_hrs', 'email_open_rate', 'nps_score',
       'tenure_days', 'gender_Female', 'gender_Male', 'gender_Other',
       'region_Cork', 'region_Dublin', 'region_Galway', 'region_Other',
       'customer_type_Business', 'customer_type_Family',
       'customer_type_Individual', 'plan_type_Postpay',
       'plan_type_Prepay', 'plan_type_SIM-only', 'plan_tier_Essentials',
       'plan_tier_Freedom', 'plan_tier_Unlimited',
       'payment_method_Credit Card', 'payment_method_Debit Card',
       'payment_method_Direct Debit']
    
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[MODEL_FEATURES]



# ========== Streamlit UI ==========



col1, col2 = st.columns([6,1])
with col1:
    st.title("🚀 Playground")
    st.markdown("Here you can predict whether a customer will churn based on their engagement, behavior, and subscription data.")
with col2:
    st.image("assets/logo.png",width=300)
st.divider()

st.subheader("📋 User Input Options")
# Load your data at the top for slider ranges
slider_df = pd.read_csv("data/churn_data.csv")

def round_down_5(x):
    return int(np.floor(x / 5.0) * 5)
def round_up_5(x):
    return int(np.ceil(x / 5.0) * 5)
def round_down_5f(x):
    return float(np.floor(x / 5.0) * 5)
def round_up_5f(x):
    return float(np.ceil(x / 5.0) * 5)


# First row: Subscription Details and User Profile
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    with st.expander("📦 Subscription Details", expanded=True):
        age = st.slider('Age', round_down_5(slider_df['age'].min()), round_up_5(slider_df['age'].max()), int(slider_df['age'].mean()), key="age")
        monthly_fee = st.slider('Monthly Fee (€)', round_down_5f(slider_df['monthly_fee_€'].min()), round_up_5f(slider_df['monthly_fee_€'].max()), float(slider_df['monthly_fee_€'].mean()), key="monthly_fee")
        contract_length = st.slider('Contract Length (Months)', round_down_5(slider_df['contract_length_months'].min()), round_up_5(slider_df['contract_length_months'].max()), int(slider_df['contract_length_months'].mean()), key="contract_length")
        auto_renewal = st.selectbox('Auto Renewal Enabled?', ['Yes', 'No'], key="auto_renewal")
        discount_last_renewal = st.selectbox('Discount Used at Last Renewal?', ['Yes', 'No'], key="discount_last_renewal")
        downgraded_last_6m = st.selectbox('Downgraded in Last 6 Months?', ['Yes', 'No'], key="downgraded_last_6m")
        last_offer_accepted = st.selectbox('Last Offer Accepted?', ['Yes', 'No'], key="last_offer_accepted")
        plan_type = st.selectbox('Plan Type', ['Postpay', 'Prepay', 'SIM-only'], key="plan_type")
        plan_tier = st.selectbox('Plan Tier', ['Essentials', 'Freedom', 'Unlimited'], key="plan_tier")
        payment_method = st.selectbox('Payment Method', ['Credit Card', 'Debit Card', 'Direct Debit'], key="payment_method")
    with st.expander("👤 User Profile", expanded=True):
        gender = st.selectbox('Gender', ['Female', 'Male', 'Other'], key="gender")
        region = st.selectbox('Region', ['Cork', 'Dublin', 'Galway', 'Other'], key="region")
        customer_type = st.selectbox('Customer Type', ['Business', 'Family', 'Individual'], key="customer_type")
        loyalty_member = st.selectbox('Loyalty Member?', ['Yes', 'No'], key="loyalty_member")
        tenure_days = st.slider('Tenure (Days)', round_down_5(slider_df['tenure_days'].min()), round_up_5(slider_df['tenure_days'].max()), int(slider_df['tenure_days'].mean()), key="tenure_days")

# Second row: Engagement Metrics and Engagement Scores
with row1_col2:
    with st.expander("📊 Engagement Metrics", expanded=True):
        app_logins_per_month = st.slider('App Logins per Month', round_down_5(slider_df['app_logins_per_month'].min()), round_up_5(slider_df['app_logins_per_month'].max()), int(slider_df['app_logins_per_month'].mean()), key="app_logins_per_month")
        dropped_calls_last_month = st.slider('Dropped Calls Last Month', round_down_5(slider_df['dropped_calls_last_month'].min()), round_up_5(slider_df['dropped_calls_last_month'].max()), int(slider_df['dropped_calls_last_month'].mean()), key="dropped_calls_last_month")
        avg_download_speed_mbps = st.slider('Avg Download Speed (Mbps)', round_down_5f(slider_df['avg_download_speed_mbps'].min()), round_up_5f(slider_df['avg_download_speed_mbps'].max()), float(slider_df['avg_download_speed_mbps'].mean()), key="avg_download_speed_mbps")
        coverage_complaints_6m = st.slider('Coverage Complaints (6m)', round_down_5(slider_df['coverage_complaints_6m'].min()), round_up_5(slider_df['coverage_complaints_6m'].max()), int(slider_df['coverage_complaints_6m'].mean()), key="coverage_complaints_6m")
        support_tickets_6m = st.slider('Support Tickets (6m)', round_down_5(slider_df['support_tickets_6m'].min()), round_up_5(slider_df['support_tickets_6m'].max()), int(slider_df['support_tickets_6m'].mean()), key="support_tickets_6m")
        avg_resolution_time_hrs = st.slider('Avg Resolution Time (hrs)', round_down_5f(slider_df['avg_resolution_time_hrs'].min()), round_up_5f(slider_df['avg_resolution_time_hrs'].max()), float(slider_df['avg_resolution_time_hrs'].mean()), key="avg_resolution_time_hrs")
        sms_per_month = st.slider('SMS per Month', round_down_5(slider_df['sms_per_month'].min()), round_up_5(slider_df['sms_per_month'].max()), int(slider_df['sms_per_month'].mean()), key="sms_per_month")
        avg_monthly_data_gb = st.slider('Avg Monthly Data (GB)', round_down_5f(slider_df['avg_monthly_data_gb'].min()), round_up_5f(slider_df['avg_monthly_data_gb'].max()), float(slider_df['avg_monthly_data_gb'].mean()), key="avg_monthly_data_gb")
        avg_monthly_voice_minutes = st.slider('Avg Monthly Voice Minutes', round_down_5f(slider_df['avg_monthly_voice_minutes'].min()), round_up_5f(slider_df['avg_monthly_voice_minutes'].max()), float(slider_df['avg_monthly_voice_minutes'].mean()), key="avg_monthly_voice_minutes")
        roaming_last_3m = st.selectbox('Roaming in Last 3 Months?', ['Yes', 'No'], key="roaming_last_3m")
        on_time_payments = st.selectbox('On Time Payments?', ['Yes', 'No'], key="on_time_payments")
        late_payments_6m = st.slider('Late Payments (6m)', round_down_5(slider_df['late_payments_6m'].min()), round_up_5(slider_df['late_payments_6m'].max()), int(slider_df['late_payments_6m'].mean()), key="late_payments_6m")
        outstanding_balance = st.slider('Outstanding Balance (€)', round_down_5f(slider_df['outstanding_balance_€'].min()), round_up_5f(slider_df['outstanding_balance_€'].max()), float(slider_df['outstanding_balance_€'].mean()), key="outstanding_balance")
    with st.expander("📈 Engagement Scores", expanded=True):
        csat_score = st.slider('CSAT Score (1-5)', round_down_5(slider_df['csat_score'].min()), round_up_5(slider_df['csat_score'].max()), int(slider_df['csat_score'].mean()), key="csat_score")
        campaign_ctr = st.slider('Campaign CTR', round_down_5f(slider_df['campaign_ctr'].min()), round_up_5f(slider_df['campaign_ctr'].max()), float(slider_df['campaign_ctr'].mean()), 0.01, key="campaign_ctr")
        email_open_rate = st.slider('Email Open Rate', round_down_5f(slider_df['email_open_rate'].min()), round_up_5f(slider_df['email_open_rate'].max()), float(slider_df['email_open_rate'].mean()), 0.01, key="email_open_rate")
        nps_score = st.slider('NPS Score', round_down_5(slider_df['nps_score'].min()), round_up_5(slider_df['nps_score'].max()), int(slider_df['nps_score'].mean()), key="nps_score")
        

col5, col6 = st.columns([1, 1])
predict_button = col5.button("🔍 Predict", use_container_width=True)
clear_button = col6.button("🗑️ Clear Inputs", use_container_width=True)

user_input = {
    'age': age,
    'monthly_fee_€': monthly_fee,
    'contract_length_months': contract_length,
    'auto_renewal': auto_renewal,
    'discount_last_renewal': discount_last_renewal,
    'downgraded_last_6m': downgraded_last_6m,
    'last_offer_accepted': last_offer_accepted,
    'plan_type': plan_type,
    'plan_tier': plan_tier,
    'payment_method': payment_method,
    # User Profile
    'gender': gender,
    'region': region,
    'customer_type': customer_type,
    # Engagement Metrics
    'app_logins_per_month': app_logins_per_month,
    'dropped_calls_last_month': dropped_calls_last_month,
    'avg_download_speed_mbps': avg_download_speed_mbps,
    'coverage_complaints_6m': coverage_complaints_6m,
    'support_tickets_6m': support_tickets_6m,
    'avg_resolution_time_hrs': avg_resolution_time_hrs,
    'sms_per_month': sms_per_month,
    'avg_monthly_data_gb': avg_monthly_data_gb,
    'avg_monthly_voice_minutes': avg_monthly_voice_minutes,
    'roaming_last_3m': roaming_last_3m,
    'loyalty_member': loyalty_member,
    'on_time_payments': on_time_payments,
    'late_payments_6m': late_payments_6m,
    'outstanding_balance_€': outstanding_balance,
    # Engagement Scores
    'csat_score': csat_score,
    'campaign_ctr': campaign_ctr,
    'email_open_rate': email_open_rate,
    'nps_score': nps_score,
    'tenure_days': tenure_days
}


if clear_button:
    st.rerun()

if predict_button:
    df = preprocess(user_input)
    model = load_xgb_model()
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"
    st.success(f"📊 **Prediction:** {result}")
    st.info(f"🧠 Confidence: **{proba * 100:.2f}%** for Churn")

    st.subheader("🔎 Feature Importance")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)

    # --- Waterfall Plot ---
    row = shap_values[0]
    shap_impact = row.values
    features = row.feature_names
    base_val = row.base_values

    top_idx = np.argsort(np.abs(shap_impact))[-15:]
    top_features = [features[i] for i in top_idx]
    top_shap = shap_impact[top_idx]

    fig_waterfall = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(top_features),
        x=top_shap,
        y=top_features,
        text=[f"{v:.3f}" for v in top_shap],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "red"}},
        increasing={"marker": {"color": "green"}},
    ))

    fig_waterfall.update_layout(
        title="",
        xaxis_title="SHAP Value Impact",
        yaxis_title="Feature",
        waterfallgap=0.4
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)


