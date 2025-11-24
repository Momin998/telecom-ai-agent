import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from textblob import TextBlob

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Jazz Enterprise AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA LOADING & SAVING LOGIC ---
@st.cache_data
def load_initial_data():
    try:
        df = pd.read_csv('Comcast.csv')
        # Data Cleaning
        if 'Customer Complaint' in df.columns:
            df['Customer Complaint'] = df['Customer Complaint'].str.replace('Comcast', 'Jazz', case=False)
            df['Customer Complaint'] = df['Customer Complaint'].str.replace('comcast', 'Jazz', case=False)
            df = df.rename(columns={'Customer Complaint': 'text', 'Received Via': 'category', 'Status': 'status'})
            df = df.dropna(subset=['text', 'category'])
        
        # Ensure all columns exist
        if 'Sentiment' not in df.columns: df['Sentiment'] = 'Neutral'
        if 'Ticket_ID' not in df.columns: df['Ticket_ID'] = "N/A"
        if 'Time' not in df.columns: df['Time'] = "N/A"
        if 'Phone_Number' not in df.columns: df['Phone_Number'] = "N/A"
            
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['text', 'category', 'status', 'Sentiment', 'Ticket_ID', 'Time', 'Phone_Number'])

if 'df' not in st.session_state:
    st.session_state['df'] = load_initial_data()

df = st.session_state['df']

# --- 3. BIG DATA TRAINING (FULL LIST) ---
big_training_data = [
    # BILLING
    {"text": "I have been overcharged this month", "category": "Billing", "status": "Open"},
    {"text": "My balance was deducted automatically", "category": "Billing", "status": "Open"},
    {"text": "Incorrect billing amount", "category": "Billing", "status": "Pending"},
    {"text": "Please refund my money", "category": "Billing", "status": "Open"},
    {"text": "Where did my recharge go?", "category": "Billing", "status": "Open"},
    {"text": "Hidden charges applied", "category": "Billing", "status": "Solved"},
    {"text": "I paid the bill but service not active", "category": "Billing", "status": "Open"},
    {"text": "Tax deduction is too high", "category": "Billing", "status": "Pending"},
    {"text": "Package not subscribed money cut", "category": "Billing", "status": "Open"},
    {"text": "Requesting cashback refund", "category": "Billing", "status": "Open"},
    {"text": "Bill history wrong calculations", "category": "Billing", "status": "Pending"},
    {"text": "Dispute charge on bill", "category": "Billing", "status": "Open"},
    
    # CUSTOMER SUPPORT / HUMAN
    {"text": "I want to talk to a human agent", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Connect me to customer support", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Speak to representative", "category": "Customer Care Call", "status": "Open"},
    {"text": "Is there a real person?", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Connect call to staff immediately", "category": "Customer Care Call", "status": "Closed"},
    {"text": "I want to speak to a manager", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Helpline not working connect chat", "category": "Customer Care Call", "status": "Open"},
    {"text": "Transfer to live agent", "category": "Customer Care Call", "status": "Pending"},
    {"text": "Change ownership of SIM", "category": "Customer Care Call", "status": "Solved"},
    {"text": "My SIM is blocked", "category": "Customer Care Call", "status": "Open"},
    {"text": "I lost my SIM card", "category": "Customer Care Call", "status": "Solved"},
    {"text": "PUK code required", "category": "Customer Care Call", "status": "Open"},
    {"text": "Complaint about service", "category": "Customer Care Call", "status": "Open"},

    # INTERNET
    {"text": "My internet speed is extremely slow", "category": "Internet", "status": "Open"},
    {"text": "Buffering issues YouTube", "category": "Internet", "status": "Open"},
    {"text": "Packet loss high ping", "category": "Internet", "status": "Pending"},
    {"text": "Router blinking red light", "category": "Internet", "status": "Open"},
    {"text": "No internet access connected", "category": "Internet", "status": "Open"},
    {"text": "WiFi signal weak", "category": "Internet", "status": "Solved"},
    {"text": "4G LTE not working", "category": "Internet", "status": "Open"},
    {"text": "Data is not working", "category": "Internet", "status": "Open"},
]

# Force Training
df_extra = pd.DataFrame(big_training_data * 20)
df_train = pd.concat([df, df_extra], ignore_index=True)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df_train['text'], df_train['category'])

# --- 4. SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Jazz_logo.png/320px-Jazz_logo.png", width=120)
st.sidebar.markdown("### 🎛️ System Access")
user_role = st.sidebar.radio("Select User Mode:", ["Customer Portal", "Manager Dashboard"])
st.sidebar.markdown("---")

# --- 5. MAIN LOGIC ---

# =========================================
# SCENARIO A: CUSTOMER VIEW
# =========================================
if user_role == "Customer Portal":
    
    st.title("👋 Jazz Customer Support AI")
    
    # LOGIN GATE
    if 'customer_logged_in' not in st.session_state:
        st.session_state['customer_logged_in'] = False

    if not st.session_state['customer_logged_in']:
        st.info("🔒 Please verify your identity.")
        with st.form("login_form"):
            phone = st.text_input("Enter Registered Phone Number:", placeholder="03001234567")
            submitted = st.form_submit_button("🔓 Start Chat")
            if submitted:
                if len(phone) == 11 and phone.startswith("03"):
                    st.session_state['customer_logged_in'] = True
                    st.session_state['phone'] = phone
                    st.success("Verified!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid Number format.")
    else:
        st.caption(f"Logged in as: {st.session_state['phone']}")
        if st.button("Log Out"):
            st.session_state['customer_logged_in'] = False
            st.rerun()
        st.markdown("---")
        
        # CHAT INTERFACE
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type complaint here..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            prompt_lower = prompt.lower()
            greetings = ["hello", "hi", "hey", "salam"]
            thanks = ["thank", "thanks", "ok", "solved"]
            
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    time.sleep(0.8)
                    
                    if any(x in prompt_lower for x in greetings):
                        response = "👋 **Hello!** I am your Jazz AI Assistant."
                    elif any(x in prompt_lower for x in thanks):
                        response = "🌟 **You're Welcome!** Happy to help."
                        st.balloons()
                    else:
                        category = model.predict([prompt])[0]
                        blob = TextBlob(prompt)
                        mood = "Negative" if blob.sentiment.polarity < -0.1 else "Positive" if blob.sentiment.polarity > 0.1 else "Neutral"
                        mood_icon = "😡" if mood == "Negative" else "😊" if mood == "Positive" else "😐"
                        
                        response = f"**Category:** {category} | **Mood:** {mood} {mood_icon}\n\n"
                        if "Internet" in category: response += "💡 **Solution:** Restart Router (30s off/on)."
                        elif "Billing" in category: response += "💡 **Solution:** Check Jazz World App."
                        elif "Customer Care Call" in category: response += "🎧 Connecting to Live Agent..."
                        else: response += "💡 **Solution:** Ticket logged."
                        
                        st.session_state['show_buttons'] = True
                        st.session_state['last_cat'] = category
                        st.session_state['last_mood'] = mood
                        st.session_state['last_txt'] = prompt

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

        # BUTTONS
        if st.session_state.get('show_buttons', False):
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            if c1.button("✅ Solved"):
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_row = {
                    'text': st.session_state['last_txt'], 
                    'category': st.session_state['last_cat'], 
                    'status': 'Solved', 
                    'Sentiment': st.session_state['last_mood'],
                    'Ticket_ID': "Auto-Resolved",
                    'Time': current_time,
                    'Phone_Number': st.session_state['phone']
                }
                updated_df = pd.concat([st.session_state['df'], pd.DataFrame([new_row])], ignore_index=True)
                st.session_state['df'] = updated_df
                updated_df.to_csv('Comcast.csv', index=False)
                st.success("Saved!")
                st.session_state['show_buttons'] = False
                time.sleep(1)
                st.rerun()
            
            if c2.button("❌ Escalate"):
                st.session_state['show_form'] = True
        
        # MANAGER FORM
        if st.session_state.get('show_form', False):
            with st.form("esc_form"):
                st.write("⚠️ Escalating to Human Manager")
                ph = st.text_input("Confirm Phone:", value=st.session_state.get('phone', ''))
                sub = st.form_submit_button("Send to Manager")
                if sub:
                    import random
                    ticket_id = random.randint(1000, 9999)
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    new_row = {
                        'text': st.session_state['last_txt'], 
                        'category': st.session_state['last_cat'], 
                        'status': 'Escalated', 
                        'Sentiment': st.session_state['last_mood'],
                        'Ticket_ID': ticket_id,
                        'Time': current_time,
                        'Phone_Number': ph
                    }
                    updated_df = pd.concat([st.session_state['df'], pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state['df'] = updated_df
                    updated_df.to_csv('Comcast.csv', index=False)
                    
                    st.success(f"Escalated! Ticket #{ticket_id}")
                    st.session_state['show_buttons'] = False
                    st.session_state['show_form'] = False
                    time.sleep(1)
                    st.rerun()

# =========================================
# SCENARIO B: MANAGER VIEW
# =========================================
elif user_role == "Manager Dashboard":
    
    st.sidebar.warning("🔒 Authorization Required")
    password = st.sidebar.text_input("Admin Password:", type="password")
    
    if password == "admin123":
        st.title("📊 Executive Analytics Dashboard")
        
        filter_stat = st.sidebar.multiselect("Filter Status:", options=df['status'].unique(), default=df['status'].unique())
        df_view = df[df['status'].isin(filter_stat)]
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Tickets", len(df_view))
        k2.metric("Internet Issues", len(df_view[df_view['category'].str.contains("Internet")]))
        k3.metric("Solved", len(df_view[df_view['status'].isin(["Solved", "Closed"])]))
        k4.metric("Escalated", len(df_view[df_view['status'] == 'Escalated']), "Action Req.")
        
        st.markdown("---")
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Complaints Volume")
            v_count = df_view['category'].value_counts().reset_index()
            v_count.columns = ['Category', 'Count']
            st.plotly_chart(px.bar(v_count, x='Category', y='Count', color='Category'), use_container_width=True)
        with g2:
            st.subheader("Sentiment Analysis")
            if 'Sentiment' in df_view.columns:
                s_count = df_view['Sentiment'].value_counts().reset_index()
                s_count.columns = ['Mood', 'Count']
                st.plotly_chart(px.pie(s_count, names='Mood', values='Count', color='Mood', color_discrete_map={"Negative":"red", "Positive":"green", "Neutral":"blue"}), use_container_width=True)
        
        # --- DETAILED TABLE WITH PHONE & TIME ---
        st.markdown("### 📋 Recent Escalations (Action Required)")
        escalated_df = df[df['status'] == 'Escalated']
        if not escalated_df.empty:
            display_cols = ['Ticket_ID', 'Time', 'Phone_Number', 'category', 'text', 'Sentiment']
            existing_cols = [c for c in display_cols if c in escalated_df.columns]
            st.dataframe(escalated_df[existing_cols].tail(10), use_container_width=True)
        else:
            st.success("No Pending Escalations. Good Job! 🎉")
        
    else:
        st.error("🚫 Access Denied.")