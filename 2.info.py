import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import os

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatOpenAI(
    openai_api_base="https://integrate.api.nvidia.com/v1",
    openai_api_key=NVIDIA_API_KEY,
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1"
)

llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)

def prompt(feature_shap_importance: dict ,proba: int, customer_row): 
    system_message = f"""
                YOU ARE AN MACHINE LEARNING MODEL EXPLAINABILITY EXPERT
                Here are the details for a  is assisting:
                    - Dictioary of feature_name, shap_impact and feature_importance for xgboost machine learning model: {feature_shap_importance}
                    - Models Predicted Churn Probability: {proba}
                    - Customer Row: {customer_row}
                Based on this information, explain to the agent in non-technical terms:
                    1. Provide summary of who the customer is from user context features and his current status of action context features.
                    2. Identify the top 3 reasons for the customers potential churn. Provide a brief explanation of why these
                    features significantly influence the churn prediction. Dont inclde any technical details like shap scores, provide business context.
                    3. Suggest the top 3 actions the agent can take to reduce the likelihood of churn, based on the feature impacts. Each suggestion should include:
                        - An explanation of why this action is expected to impact churn, based solely on the data provided.
                        - Dont include any technical details like shap scores, provide business context.
                Remember :
                    - The magnitude of a SHAP value indicates the strength of a feature's influence on the prediction.
                    - Positive SHAP values increase the likelihood of churn; negative values decrease it.
                    - Feature Importances values adds up to 1, greater the value higher the feature is important in prediction.
                    - Recommendations should be strictly based on the information provided in the SHAP contributions and customer features.

                Keep the report short and concise in 2-3 paragraphs (max 150 words in total).
                Do not include any other text in the report.
                Make sure the response is in markdown format with proper use only H3, H4, H5 and emojis.
            """
    return system_message

def escape_curly_braces(s):
    return s.replace("{", "{{").replace("}", "}}")


prompt2 = ChatPromptTemplate.from_template("""
You are an offer recommendation expert.
You are given a customer profile and a list of offers.
You need to recommend the best offers for the customer.
Here is the customer profile:{customer_row}
Here is the list of offers csv:{offers}
Give the output in markdown format with only Recommendation subheading , bullet points and emojis.
Only provide the top three offer recommendation and not any other text.
""")

parser = StrOutputParser()



@st.cache_data
def load_data():
    return pd.read_csv("data/data_with_churn_score.csv")

df = load_data()

@st.cache_data
def load_offers():
    return pd.read_csv("data/offer.csv")

offers = load_offers()

# --- ML Model ---
def load_xgb_model():
    with open('model/model_2.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess(user_input):
    df = pd.DataFrame([user_input])
    df['auto_renewal'] = df['auto_renewal'].map({'Yes':1,'No':0})
    df['subscription_status'] = df['subscription_status'].map({'Active':1,'Cancelled':0})
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


# Check if the session state has the selected customer id

if 'selected_customer_id' not in st.session_state:
    st.write("No customer selected.")
else:
    customer_id = st.session_state['selected_customer_id']
    customer_row = df[df['customer_id'] == customer_id]
    if customer_row.empty:
        st.write("Customer not found.")
    else:
        customer = customer_row.iloc[0]
        st.title(f"🪪 Customer Profile for {customer['customer_id']}")    
        st.divider()

     # --- ML Prediction ---
        st.markdown("## 🤖 Churn Prediction for Customer")

        # Prepare input dict for preprocess
        input_dict = customer.to_dict()

        # Preprocess and predict
        X = preprocess(input_dict)
        model = load_xgb_model()
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]



        feature_importances = model.feature_importances_
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        row = shap_values[0]
        shap_impact = row.values
        features = row.feature_names 

        feature_shap_importance = {
            feature: {
                'shap_impact': float(shap),
                'feature_importance': float(importance)
            }
            for feature, shap, importance in zip(features, shap_impact, feature_importances)
        }

        escaped_feature_shap_importance = escape_curly_braces(str(feature_shap_importance))
        prompt_template = prompt(escaped_feature_shap_importance, proba, customer_row)
        prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_template),
                    ("human", "")
                ])
        


        col1, col2 = st.columns(2)

        with col1:
            if input_dict['churn_risk_model_1'] == "Very High":
                result = "Very High " + " " + "Risk of Churn"
                st.error(f"📊 **Prediction:** {result}")
            elif input_dict['churn_risk_model_1'] == "High":
                result = "High " + " " + "Risk of Churn"
                st.error(f"📊 **Prediction:** {result}")
            elif input_dict['churn_risk_model_1'] == "Medium":
                result = "Medium " + " " + "Risk of Churn"
                st.warning(f"📊 **Prediction:** {result}")
            elif input_dict['churn_risk_model_1'] == "Low":
                result = "Low " + " " + "Risk of Churn"
                st.success(f"📊 **Prediction:** {result}")
            else:
                st.error("📊 **Prediction:** Already Churned")    
            report_button = st.button("Get Report and Chart")
        with col2:             
            st.info(f"🧠 Model Score: **{input_dict['churn_score_model_1'] * 100:.2f}%** for Churn")

                   

        # SHAP GRAPH AND REPORT

        if report_button:
            col1, col2 = st.columns(2)
            with col1:
                with st.spinner("Generating Report..."):
                    with st.container(border=True):
                        chain = prompt | llm | parser
                        result = chain.invoke({"feature_shap_importance": escaped_feature_shap_importance, "proba": proba, "customer_row": customer_row})
                        st.markdown(f"{result}")
            with col2:
                with st.container(border=True):
                    # SHAP Feature Importance
                    st.subheader("🔎 Feature Importance")
                    top_idx = np.argsort(np.abs(shap_impact))[-10:]
                    top_features = [features[i] for i in top_idx]
                    top_shap = shap_impact[top_idx]
                    fig_waterfall = go.Figure(go.Waterfall(
                        orientation="h",
                        measure=["relative"] * len(top_features),
                        x=top_shap,
                        y=top_features,
                        text=[f"{v:.3f}" for v in top_shap],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        decreasing={"marker": {"color": "green"}},
                        increasing={"marker": {"color": "red"}},
                    ))
                    fig_waterfall.update_layout(
                        title="",
                        xaxis_title="SHAP Value Impact",
                        yaxis_title="Feature",
                        waterfallgap=0.4
                    )
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                with st.spinner("🔎 Generating Offer Recommendation..."):
                    with st.container(border=True):
                        chain2 = prompt2 | llm2 | parser
                        result2 = chain2.invoke({"customer_row": customer_row, "offers": offers})
                        st.markdown(f"{result2}")

        st.divider()

        
        # Personal Information Section
        with st.container(border=True):
            col_1, col_2, col_3 = st.columns([1.5,1.5,1])
            with col_1:
                st.subheader("👤 Personal Information")
                st.markdown(f"**🧑 Name:** {customer['first_name']} {customer['last_name']}")
                st.markdown(f"**🎂 Age:** {customer['age']}")
                st.markdown(f"**🚻 Gender:** {customer['gender']}")
            with col_2:
                if customer['subscription_status'] == "Active":
                    st.success("✅ Subscription is Active")
                else:
                    st.error("❌ Subscription is Cancelled")
                st.markdown(f"**✉️ Email:** {customer['email']}")
                st.markdown(f"**📞 Phone:** {customer['Phone']}")
                st.markdown(f"**🏠 Region:** {customer['region']}")

            with col_3:
                if customer['gender'] == "Male" or customer['gender'] == "Other":
                    st.image("assets/man.jpg",width=300)
                else:
                    st.image("assets/woman.jpg",width=300)


        
        # Subscription Details
        with st.container(border=True):
            st.subheader("📋 Subscription Details")
            col_sub1, col_sub2, col_sub3 = st.columns(3)
            with col_sub1:
                st.markdown(f"**👤 Customer Type:** {customer['customer_type']}")
                st.markdown(f"**📦 Plan Type:** {customer['plan_type']}")
                st.markdown(f"**🏷️ Plan Tier:** {customer['plan_tier']}")
                st.markdown(f"**📅 Contract Length:** {customer['contract_length_months']} months")
            with col_sub2:
                if customer['auto_renewal'] == "Yes":
                    st.markdown(f"**📦 Auto Renewal :** ✅ {customer['auto_renewal']}")
                else:
                    st.markdown(f"**📦 Auto Renewal :** ❌ {customer['auto_renewal']}")
                st.markdown(f"**📅 Start Date:** {customer['subscription_start_date']}")
                if customer['subscription_status'] == "Cancelled":
                    st.markdown(f"**❌ End Date:** {customer['subscription_end_date']}")
                if customer['loyalty_member'] == "Yes":
                    st.markdown(f"**👑 Loyalty Member :** ✅ {customer['loyalty_member']}")
                else:
                    st.markdown(f"**👑 Loyalty Member :** ❌ {customer['loyalty_member']}")                    
                if customer['last_offer_accepted'] == "Yes":
                    st.markdown(f"**🎁 Last Offer Accepted :** ✅ {customer['last_offer_accepted']}")
                else:
                    st.markdown(f"**🎁 Last Offer Accepted :** ❌ {customer['last_offer_accepted']}") 
            with col_sub3:
                st.markdown(f"**💳 Payment Method:** {customer['payment_method']}")
                if customer['on_time_payments'] == "Yes":
                    st.markdown(f"**💰 On-time Payments :** ✅ {customer['on_time_payments']}")
                else:
                    st.markdown(f"**💰 On-time Payments :** ❌ {customer['on_time_payments']}")
                if customer['discount_last_renewal'] == "Yes":
                    st.markdown(f"**🏷️ Discount Used :** ✅ {customer['discount_last_renewal']}")
                else:
                    st.markdown(f"**🏷️ Discount Used :** ❌ {customer['discount_last_renewal']}")
                if customer['downgraded_last_6m'] == "Yes":
                    st.markdown(f"**⬇️ Downgraded (6m) :** ⚠️ {customer['downgraded_last_6m']}")
                else:
                    st.markdown(f"**⬇️ Downgraded (6m) :** ✅ {customer['downgraded_last_6m']}")

        # Payment Details
        with st.container(border=True):
            st.subheader("💳 Payment Details")
            col_pay1, col_pay2, col_pay3, col_pay4 = st.columns(4)
            with col_pay1:
                st.metric("💰 Monthly Fee", f"€{customer['monthly_fee_€']}")
            with col_pay2:
                st.metric("💰 Outstanding Balance", f"€{customer['outstanding_balance_€']:.2f}")
            with col_pay3:
                st.metric("⏱️ Tenure", f"{customer['tenure_days']} days")
            with col_pay4:
                st.metric("⏰ Late Payments (6m)", customer['late_payments_6m'])
        
        colo1, colo2 = st.columns(2)

        with colo1:
        # Usage
            with st.container(border=True):
                st.subheader("📱 Usage")
                col_use1, col_use2 = st.columns(2)
                with col_use1:
                    st.metric("📊 Avg Data/Month", f"{customer['avg_monthly_data_gb']} GB")
                    st.metric("📞 Avg Voice/Min", customer['avg_monthly_voice_minutes'])
                    st.metric("💬 SMS/Month", customer['sms_per_month'])
                    if customer['roaming_last_3m'] == "Yes":
                        st.markdown(f"**🌍 Roaming (3m) :** ✅ {customer['roaming_last_3m']}")
                    else:
                        st.markdown(f"**🌍 Roaming (3m) :** ❌ {customer['roaming_last_3m']}")
                with col_use2:
                    st.metric("📱 App Logins/Month", customer['app_logins_per_month'])
                    st.metric("📶 Download Speed", f"{customer['avg_download_speed_mbps']} Mbps")
                    st.metric("📞 Dropped Calls", customer['dropped_calls_last_month'])
                    st.metric("📡 Coverage Complaints (6m)", customer['coverage_complaints_6m'])
        with colo2:
            # Customer Service
            with st.container(border=True):
                st.subheader("🛠️ Engagement & Service")
                col_cs1, col_cs2 = st.columns(2)
                with col_cs1:
                    st.metric("🎟️ Support Tickets (6m)", customer['support_tickets_6m'])
                    st.metric("⏱️ Avg Resolution Time", f"{customer['avg_resolution_time_hrs']} hrs")
                    st.metric("📈 NPS Score", customer['nps_score'])
                with col_cs2:
                    st.metric("📧 Email Open Rate", f"{customer['email_open_rate']:.1%}")   
                    st.metric("😊 CSAT Score", f"{customer['csat_score']}/5")
                    st.metric("📊 Campaign CTR", f"{customer['campaign_ctr']:.1%}")

    st.divider()

    colm5, colm6 = st.columns(2)

    with colm5:
        with st.container(border=True):
            st.markdown("#### 📊 Email & Campaign Performance")
            # Generate realistic email and campaign data for 30 days
            days = [f"Day {i+1}" for i in range(30)]
            
            # Email open rate (0-1 range) - realistic email engagement
            email_open_rate = np.random.normal(0.35, 0.1, 30)  # Mean 35%, std 10%
            email_open_rate = np.clip(email_open_rate, 0.05, 0.8)  # Clip between 5% and 80%
            
            # Campaign CTR (0-1 range) - realistic click-through rates
            campaign_ctr = np.random.normal(0.08, 0.03, 30)  # Mean 8%, std 3%
            campaign_ctr = np.clip(campaign_ctr, 0.01, 0.25)  # Clip between 1% and 25%
            
            # Create multi-line chart for email and campaign metrics
            fig_email_campaign = go.Figure()
            
            fig_email_campaign.add_trace(go.Scatter(
                x=days, y=email_open_rate,
                mode='lines+markers',
                name='Email Open Rate',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            fig_email_campaign.add_trace(go.Scatter(
                x=days, y=campaign_ctr,
                mode='lines+markers',
                name='Campaign CTR',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=4)
            ))
            
            fig_email_campaign.update_layout(
                title="Email & Campaign Performance (Last 30 Days)",
                xaxis_title="Day",
                yaxis_title="Rate (0-1)",
                yaxis=dict(range=[0, 1]),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400
            )
            
            st.plotly_chart(fig_email_campaign, use_container_width=True)

    with colm6:
        with st.container(border=True):
            st.markdown("#### 📈 Customer Satisfaction Scores")
            # Generate realistic satisfaction scores for 30 days
            days = [f"Day {i+1}" for i in range(30)]
            
            # CSAT score (1-5 range) - customer satisfaction
            csat_score = np.random.normal(3.8, 0.5, 30)  # Mean 3.8, std 0.5
            csat_score = np.clip(csat_score, 1.0, 5.0)  # Clip between 1 and 5
            
            # NPS score (-5 to 5 range) - net promoter score
            nps_score = np.random.normal(0.5, 1.5, 30)  # Mean 0.5, std 1.5
            nps_score = np.clip(nps_score, -5.0, 5.0)  # Clip between -5 and 5
            
            # Create multi-line chart for satisfaction scores
            fig_satisfaction = go.Figure()
            
            fig_satisfaction.add_trace(go.Scatter(
                x=days, y=csat_score,
                mode='lines+markers',
                name='CSAT Score',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4)
            ))
            
            fig_satisfaction.add_trace(go.Scatter(
                x=days, y=nps_score,
                mode='lines+markers',
                name='NPS Score',
                line=dict(color='#d62728', width=2),
                marker=dict(size=4)
            ))
            
            fig_satisfaction.update_layout(
                title="Customer Satisfaction Trends (Last 30 Days)",
                xaxis_title="Day",
                yaxis_title="Score",
                yaxis=dict(range=[-5, 5]),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400
            )
            
            st.plotly_chart(fig_satisfaction, use_container_width=True)

    # New section for daily usage analytics
    with st.container(border=True):
        st.markdown("#### 📊 Daily Usage Analytics")
        # Generate realistic daily usage data for 30 days
        days = [f"Day {i+1}" for i in range(30)]
        
        # Internet data usage (GB per day) - realistic mobile usage
        data_usage = np.random.normal(2.5, 0.8, 30)  # Mean 2.5GB, std 0.8GB
        data_usage = np.clip(data_usage, 0.1, 5.0)  # Clip between 0.1 and 5GB
        
        # Voice minutes per day - realistic call patterns
        voice_minutes = np.random.normal(45, 15, 30)  # Mean 45 min, std 15 min
        voice_minutes = np.clip(voice_minutes, 0, 120)  # Clip between 0 and 120 min
        
        # SMS usage per day - declining trend
        sms_count = np.random.normal(8, 3, 30)  # Mean 8 SMS, std 3
        sms_count = np.clip(sms_count, 0, 20)  # Clip between 0 and 20
        
        # Download speed (Mbps) - variable speeds
        download_speed = np.random.normal(50, 15, 30)  # Mean 50 Mbps, std 15
        download_speed = np.clip(download_speed, 10, 100)  # Clip between 10 and 100 Mbps
        
        # Create multi-line chart
        fig_usage = go.Figure()
        
        fig_usage.add_trace(go.Scatter(
            x=days, y=data_usage,
            mode='lines+markers',
            name='Data Usage (GB)',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        fig_usage.add_trace(go.Scatter(
            x=days, y=voice_minutes,
            mode='lines+markers',
            name='Voice Minutes',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4)
        ))
        
        fig_usage.add_trace(go.Scatter(
            x=days, y=sms_count,
            mode='lines+markers',
            name='SMS Count',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=4)
        ))
        
        fig_usage.add_trace(go.Scatter(
            x=days, y=download_speed,
            mode='lines+markers',
            name='Download Speed (Mbps)',
            line=dict(color='#d62728', width=2),
            marker=dict(size=4)
        ))
        
        fig_usage.update_layout(
            title="Daily Usage Patterns (Last 30 Days)",
            xaxis_title="Day",
            yaxis_title="Usage",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        st.plotly_chart(fig_usage, use_container_width=True)

    st.divider()
   
