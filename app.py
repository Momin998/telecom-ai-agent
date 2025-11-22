import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Jazz Enterprise AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE & DATA LOADING ---
if 'df' not in st.session_state:
    try:
        df = pd.read_csv('Comcast.csv')
        df['Customer Complaint'] = df['Customer Complaint'].str.replace('Comcast', 'Jazz', case=False)
        df['Customer Complaint'] = df['Customer Complaint'].str.replace('comcast', 'Jazz', case=False)
        df = df.rename(columns={'Customer Complaint': 'text', 'Received Via': 'category', 'Status': 'status'})
        df = df.dropna(subset=['text', 'category'])
        st.session_state['df'] = df
    except FileNotFoundError:
        st.error("❌ Error: 'Comcast.csv' file not found.")
        st.session_state['df'] = pd.DataFrame(columns=['text', 'category', 'status'])

df = st.session_state['df']

# --- 3. BIG DATA TRAINING (Advanced English Logic) ---
big_training_data = [
    # BILLING ISSUES
    {"text": "I have been overcharged this month", "category": "Billing", "status": "Open"},
    {"text": "My balance was deducted automatically without any reason", "category": "Billing", "status": "Open"},
    {"text": "Incorrect billing amount on my invoice", "category": "Billing", "status": "Pending"},
    {"text": "Please refund my money immediately", "category": "Billing", "status": "Open"},
    {"text": "Where did my recharge go? Balance is zero", "category": "Billing", "status": "Open"},
    {"text": "Hidden charges applied to my account", "category": "Billing", "status": "Solved"},
    {"text": "I paid the bill but service is not active", "category": "Billing", "status": "Open"},
    {"text": "Tax deduction is too high on my recharge", "category": "Billing", "status": "Pending"},
    {"text": "My package was not subscribed but money cut", "category": "Billing", "status": "Open"},
    {"text": "Requesting a cashback or refund for failed transaction", "category": "Billing", "status": "Open"},
    {"text": "Bill history is showing wrong calculations", "category": "Billing", "status": "Pending"},
    {"text": "I want to dispute a charge on my bill", "category": "Billing", "status": "Open"},
    {"text": "My credit limit is incorrect", "category": "Billing", "status": "Solved"},
    {"text": "I was charged twice for the same package", "category": "Billing", "status": "Open"},

    # CUSTOMER SUPPORT / HUMAN AGENT
    {"text": "I want to talk to a human agent", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Connect me to customer support staff", "category": "Customer Care Call", "status": "Pending"},
    {"text": "I need to speak to a representative regarding my issue", "category": "Customer Care Call", "status": "Open"},
    {"text": "Is there a real person I can talk to?", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Connect call to staff immediately", "category": "Customer Care Call", "status": "Closed"},
    {"text": "I want to speak to a manager", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Helpline is not working, connect me to chat support", "category": "Customer Care Call", "status": "Open"},
    {"text": "Transfer this chat to a live agent right now", "category": "Customer Care Call", "status": "Pending"},
    {"text": "I need to change ownership of my SIM", "category": "Customer Care Call", "status": "Solved"},
    {"text": "My SIM is blocked, I need help from staff", "category": "Customer Care Call", "status": "Open"},
    {"text": "I lost my SIM card, please block it", "category": "Customer Care Call", "status": "Solved"},
    {"text": "PUK code required for my number", "category": "Customer Care Call", "status": "Open"},
    {"text": "I have a complaint about your service", "category": "Customer Care Call", "status": "Open"},

    # INTERNET ISSUES
    {"text": "My internet speed is extremely slow", "category": "Internet", "status": "Open"},
    {"text": "Buffering issues while watching YouTube", "category": "Internet", "status": "Open"},
    {"text": "Packet loss and high ping in games", "category": "Internet", "status": "Pending"},
    {"text": "My router is blinking red light", "category": "Internet", "status": "Open"},
    {"text": "No internet access even though connected", "category": "Internet", "status": "Open"},
    {"text": "WiFi signal is very weak in my room", "category": "Internet", "status": "Solved"},
    {"text": "4G LTE is not working on my phone", "category": "Internet", "status": "Open"},
    {"text": "Data is not working", "category": "Internet", "status": "Open"},
]

# Force Training (20x Repetition)
df_extra = pd.DataFrame(big_training_data * 20)
df_train = pd.concat([df, df_extra], ignore_index=True)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df_train['text'], df_train['category'])

# --- 4. SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Jazz_logo.png/320px-Jazz_logo.png", width=120)
st.sidebar.markdown("### ⚙️ Live Controls")

status_filter = st.sidebar.multiselect(
    "Filter Status:",
    options=df['status'].unique(),
    default=df['status'].unique()
)
df_filtered = df[df['status'].isin(status_filter)]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 **Active Tickets:** {len(df)}")

# --- 5. MAIN TABS ---
st.title("📡 Jazz Enterprise Solutions")
tab1, tab2 = st.tabs(["📊 Live Dashboard", "🤖 AI Resolution Agent"])

# === TAB 1: DASHBOARD ===
with tab1:
    st.markdown("### 📈 Real-Time Analytics")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total = len(df_filtered)
    internet = len(df_filtered[df_filtered['category'].str.contains("Internet")])
    solved = len(df_filtered[df_filtered['status'].isin(["Closed", "Solved"])])
    escalated = len(df_filtered[df_filtered['status'].str.contains("Escalated", na=False)]) 
    
    kpi1.metric("Total Tickets", total, "Live")
    kpi2.metric("Internet Issues", internet, "Top Category")
    kpi3.metric("Solved Cases", solved, "✅ Success")
    kpi4.metric("Escalated (Manager)", escalated, "🔥 High Priority")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📉 Complaints by Category")
        cat_counts = df_filtered['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig_bar = px.bar(cat_counts, x='Category', y='Count', color='Category', template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("🥧 Live Status Distribution")
        fig_pie = px.pie(df_filtered, names='status', hole=0.4, template="plotly_white", title="Current Status")
        st.plotly_chart(fig_pie, use_container_width=True)

# === TAB 2: AI AGENT ===
with tab2:
    st.markdown("### 🤖 Smart Agent with Feedback Loop")
    
    with st.container():
        user_input = st.text_area("📝 Enter Complaint:", height=80, placeholder="e.g., My internet is extremely slow...")
        
        if st.button("🚀 Analyze", type="primary"):
            if user_input and model:
                with st.spinner('Analyzing intent...'):
                    time.sleep(1)
                category = model.predict([user_input])[0]
                st.session_state['last_result'] = category
                st.session_state['last_input'] = user_input
                st.session_state['show_feedback'] = True
            else:
                st.warning("Please enter text first.")

        if 'show_feedback' in st.session_state and st.session_state['show_feedback']:
            
            # Bug Fix Check: Text Changed?
            if user_input != st.session_state['last_input']:
                st.warning("⚠️ You have changed the text. Please click '🚀 Analyze' again to get the new result.")
            
            else:
                category = st.session_state['last_result']
                txt_input = st.session_state['last_input']
                
                st.info(f"**AI Result:** {category}")
                
                # Customer Facing Solutions
                st.markdown("### 💡 Suggested Solution:")
                if "Internet" in category:
                     st.info("👉 **Step 1:** Please turn off your Router/Modem.\n\n👉 **Step 2:** Wait for 30 seconds and turn it on again.")
                elif "Billing" in category:
                    st.info("👉 We have checked your account history.\n\nPlease open your **Jazz World App** to view the last 3 deductions.")
                elif "Customer Care Call" in category:
                     st.info("👉 Our lines are currently busy, but we can connect you to a Live Agent via Chat immediately.")
                else:
                    st.info("👉 Your request has been noted. Our technical team will analyze this issue.")

                st.markdown("---")
                st.write("### ❓ Did this solve the issue?")
                
                c_yes, c_no = st.columns(2)
                
                # YES Button
                if c_yes.button("✅ Yes, Solved"):
                    new_row = {'text': txt_input, 'category': category, 'status': 'Solved'}
                    st.session_state['df'] = pd.concat([st.session_state['df'], pd.DataFrame([new_row])], ignore_index=True)
                    st.balloons()
                    st.success("Ticket Closed! Dashboard Updated.")
                    time.sleep(1)
                    st.rerun()
                
                # NO Button
                if c_no.button("❌ No, Issue Persists"):
                    st.session_state['show_contact_form'] = True
                
                # Escalation Form
                if 'show_contact_form' in st.session_state and st.session_state['show_contact_form']:
                    st.warning("⚠️ Escalating to Human Manager.")
                    with st.form("escalation"):
                        phone = st.text_input("Enter Phone Number:")
                        submit = st.form_submit_button("📩 Send to Manager")
                        
                        if submit:
                            ticket_id = random.randint(1000, 9999)
                            new_row = {'text': txt_input, 'category': category, 'status': 'Escalated'}
                            st.session_state['df'] = pd.concat([st.session_state['df'], pd.DataFrame([new_row])], ignore_index=True)
                            st.success(f"Request Sent! Ticket #{ticket_id} Created.")
                            time.sleep(1)
                            st.rerun()