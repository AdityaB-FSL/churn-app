import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/data_with_churn_score.csv")

df = load_data()
col1, col2 = st.columns([6,1])
with col1:
    st.title("🏠 3Ireland Risk Tracking Home")
with col2:
    st.image("assets/logo.png",width=600)
st.divider()


# Dropdowns for filtering
regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
customer_types = ["All"] + sorted(df["customer_type"].dropna().unique().tolist())
plan_types = ["All"] + sorted(df["plan_type"].dropna().unique().tolist())
plan_tiers = ["All"] + sorted(df["plan_tier"].dropna().unique().tolist())
subscription_status = ["All"] + df["subscription_status"].dropna().unique().tolist()
churn_risks = ["All"] + df["churn_risk_model_1"].dropna().unique().tolist()



st.markdown("#### 🔍 Filters:")
colf1, colf2, colf3= st.columns(3)
with colf1:
    customer_type_filter = st.segmented_control(label="Customer Type", options=customer_types, selection_mode="single", default="All")
with colf2:
    region_filter = st.segmented_control(label="Region", options=regions, selection_mode="single", default="All")
with colf3:
    plan_tier_filter = st.segmented_control(label="Plan Tier", options=plan_tiers, selection_mode="single", default="All")
    
colf4, colf5,colf6 = st.columns(3)
with colf4:
    plan_type_filter = st.segmented_control(label="Plan Type", options=plan_types, selection_mode="single", default="All")
    
with colf5:
    churn_risk_filter = st.segmented_control(label="Churn", options=churn_risks, selection_mode="single", default="All")
with colf6:
    subscription_status_filter = st.segmented_control(label="Subscription Status", options=subscription_status, selection_mode="single", default="All")

st.divider()

@st.cache_data
def get_filtered_data(df, region_filter, customer_type_filter, plan_type_filter, plan_tier_filter, churn_risk_filter, subscription_status_filter):
    filtered = df.copy()
    if region_filter != "All":
        filtered = filtered[filtered["region"] == region_filter]
    if customer_type_filter != "All":
        filtered = filtered[filtered["customer_type"] == customer_type_filter]
    if plan_type_filter != "All":
        filtered = filtered[filtered["plan_type"] == plan_type_filter]
    if plan_tier_filter != "All":
        filtered = filtered[filtered["plan_tier"] == plan_tier_filter]
    if churn_risk_filter != "All":
        filtered = filtered[filtered["churn_risk_model_1"] == churn_risk_filter]     
    if subscription_status_filter != "All":
        filtered = filtered[filtered["subscription_status"] == subscription_status_filter]  
    return filtered


filtered = get_filtered_data(df, region_filter, customer_type_filter, plan_type_filter, plan_tier_filter, churn_risk_filter, subscription_status_filter)



col1, col2, col3 = st.columns(3)
with col1:
    # Expanders for overview and subscription status
    with st.container(border=True):
        st.markdown("#### 📊 Customer Overview")
        total_customers = 1000
        filtered_customers = len(filtered)
        filtered_pct = (filtered_customers / total_customers) * 100
        colm1, colm2= st.columns(2)
        with colm1:
            st.metric(label="👥 Total Customers", value=total_customers)
        with colm2:
            st.metric(label="🔎 Filtered Customers", value=filtered_customers)
            

with col2:
    with st.container(border=True):
        st.markdown("#### 📈 Membership Distribution")
        status_counts = filtered["subscription_status"].value_counts()
        total = len(filtered)
        if total > 0:
            active = status_counts.get("Active", 0)
            cancelled = status_counts.get("Cancelled", 0)
            active_pct = (active / total) * 100
            cancelled_pct = (cancelled / total) * 100
            colm1, colm2 = st.columns(2)
            with colm1:
                st.metric(label="✅ Active", value=active)
            with colm2:
                st.metric(label="❌ Cancelled", value=cancelled)
        else:
            st.write("No customers to show distribution.")
with col3:
    with st.container(border=True):
        st.markdown("#### 🔥 Churn Risk Metrics")
        churn_counts = filtered['churn_risk_model_1'].value_counts()
        total = len(filtered)
        high = churn_counts.get('High', 0)
        medium = churn_counts.get('Medium', 0)
        low = churn_counts.get('Low', 0)
        high_pct = f"{(high/total*100):.1f}%" if total > 0 else "0.0%"
        medium_pct = f"{(medium/total*100):.1f}%" if total > 0 else "0.0%"
        low_pct = f"{(low/total*100):.1f}%" if total > 0 else "0.0%"
        colh, colm, coll = st.columns(3)
        with colh:
            st.metric(label="🔴 High Risk", value=high_pct)
        with colm:
            st.metric(label="🟠 Medium Risk", value=medium_pct)
        with coll:
            st.metric(label="🟢 Low Risk", value=low_pct)





# Customers per page dropdown
if 'per_page' not in st.session_state:
    st.session_state['per_page'] = 20

col1, col2 = st.columns([7,1])
with col1:
    search_id = st.text_input("🔍 Search by Name", "", key="full_name")
with col2:
    per_page = st.selectbox(
        "Customers per page",
        [10, 20, 50],
        index=[10, 20, 50].index(st.session_state['per_page']),
        key="per_page_select"
    )
    if per_page != st.session_state['per_page']:
        st.session_state['per_page'] = per_page
        st.session_state['page'] = 1
        st.rerun()

# Filter by search
if search_id.strip():
    search_str = search_id.strip().lower()
    filtered_page = filtered[
        filtered['first_name'].str.lower().str.contains(search_str, na=False) |
        filtered['last_name'].str.lower().str.contains(search_str, na=False) |
        filtered['full_name'].str.lower().str.contains(search_str, na=False)
    ]
    show_pagination = False
else:
    # Pagination setup
    CUSTOMERS_PER_PAGE = st.session_state['per_page']
    num_customers = len(filtered)
    num_pages = (num_customers - 1) // CUSTOMERS_PER_PAGE + 1 if num_customers > 0 else 1
    # Initialize page in session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 1
    page = st.session_state['page']
    start_idx = (page - 1) * CUSTOMERS_PER_PAGE
    end_idx = start_idx + CUSTOMERS_PER_PAGE
    filtered_page = filtered.iloc[start_idx:end_idx]
    show_pagination = True

# Show customer info in containers with clickable button
for idx, row in filtered_page.iterrows():
    with st.container(border=True):
        st.markdown(f"##### 🗂️ Customer ID: {row['customer_id']}")
        col1, col2, col3, col4 = st.columns([0.75,1,1,0.5])
        with col1:
            if row['gender'] == "Male" or row['gender'] == "Other":
                st.image("assets/man.jpg",width=150)
            else:
                st.image("assets/woman.jpg",width=150)
        with col2:
            st.markdown(f"###### **🧑 Name:** {row['first_name']} {row['last_name']}")
            st.markdown(f"###### **🆔 Contact:** {row['Phone']}")
            st.markdown(f"###### **✉️ Email:** {row['email']}")
        with col3:
            st.markdown(f"###### **🎂 Age:** {row['age']}")
            st.markdown(f"###### **🚻 Gender:** {row['gender']}")
            st.markdown(f"###### **🏠 Region:** {row['region']}")
        with col4:
            if row['subscription_status'] == "Active":
                st.badge("Active Member", icon=":material/check:", color="green")
            else:
                st.badge("Cancelled Member", icon=":material/close:", color="red")            
            info_button = st.button(label="View Info", type="primary", key=row['customer_id'], use_container_width=True)    
            if info_button:
                st.session_state['selected_customer_id'] = row['customer_id']
                st.switch_page("2.info.py")
                st.rerun()

# Pagination controls (only show if not searching)
if show_pagination:
    st.markdown("---")
    col_prev, col_page, col_next = st.columns(3)
    with col_prev:
        if page > 1:
            if st.button("⬅️ Previous", key="prev_page"):
                st.session_state['page'] = page - 1
                st.rerun()
        else:
            st.button("⬅️ Previous", key="prev_page_disabled", disabled=True)
    with col_page:
        st.markdown(f"### {page}")
    with col_next:
        if page < num_pages:
            if st.button("Next ➡️", key="next_page"):
                st.session_state['page'] = page + 1
                st.rerun()
        else:
            st.button("Next ➡️", key="next_page_disabled", disabled=True)
