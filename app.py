# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
from datetime import datetime, timedelta
# TextBlob yahan se hta diya gaya hai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# ---> 1. MODULES IMPORT (100% Intact Backend)
# ✅ UPDATE: Naye authentication functions import kiye
from db_manager import load_data_fresh, save_ticket, resolve_ticket, authenticate_customer, register_new_customer 
from ml_engine import train_bilingual_model
from ai_service import get_rag_solution, fallback_intent_classifier 

# --- 2. MODEL INITIALIZATION (CACHED) ---
@st.cache_resource
def get_cached_model():
    return train_bilingual_model()

model = get_cached_model()

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="Jazz AI Intelligence", page_icon="📡", layout="wide")

# --- 4. ENTERPRISE-GRADE UI/UX CSS OVERHAUL 🎨 ---
st.markdown("""
<style>
    /* MAIN BACKGROUND */
    .stApp {
        background: #0f172a;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    /* 🟢 SIDEBAR FIXES 🟢 */
    [data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid #1e293b !important;
    }
    
    /* Sidebar Text & Labels (Force White) */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #f8fafc !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }

    /* Radio Buttons in Sidebar */
    .stRadio p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Password & Text Inputs in Sidebar */
    [data-testid="stSidebar"] .stTextInput input {
        background: #1e293b !important;
        border: 1px solid #38bdf8 !important; 
        border-radius: 8px !important;
        color: white !important;
        padding: 8px !important;
        caret-color: #38bdf8 !important;
    }
    [data-testid="stSidebar"] .stTextInput input:focus {
        border: 1px solid #DD2476 !important;
        box-shadow: 0 0 10px rgba(221,36,118,0.5) !important;
    }

    /* MAIN TITLES */
    h1 {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #DD2476, #FF512F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }
    label p, .stMarkdown p {
        color: #94a3b8 !important;
    }

    /* THE FIX: KILL THE BOTTOM WHITE BLOCK */
    [data-testid="stBottomBlock"], .stBottom {
        background-color: transparent !important;
        background: #0f172a !important; 
    }
    
    /* 🔴 THE FIX: MAIN INPUT BOXES & BLINKING CURSOR 🔴 */
    .stTextInput input, .stChatInput textarea, .stChatInputContainer {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 10px !important;
        caret-color: #DD2476 !important;
    }
    
    .stTextInput input::placeholder, .stChatInput textarea::placeholder {
        color: #64748b !important;
    }

    .stTextInput input:focus, .stChatInput textarea:focus, .stChatInputContainer:focus-within {
        border: 2px solid #DD2476 !important;
        box-shadow: 0 0 12px rgba(221,36,118,0.6) !important;
    }

    /* CHAT BOX BUBBLES */
    [data-testid="stChatMessage"] {
        background: #1e293b !important;
        border-radius: 12px !important;
        padding: 12px !important;
        border: 1px solid #334155 !important;
        margin-bottom: 15px;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div {
        color: #ffffff !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
    }
    [data-testid="stChatMessage"]:nth-child(odd){
        border-left: 4px solid #38bdf8;
    }
    [data-testid="stChatMessage"]:nth-child(even){
        border-left: 4px solid #DD2476;
    }

    /* BUTTONS - Jazz Brand Accent */
    .stButton button, .stFormSubmitButton button {
        background: linear-gradient(135deg, #DD2476, #FF512F) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: 0.3s !important;
        width: 100%;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 8px 20px rgba(221,36,118,0.4) !important;
    }
    
    [data-testid="stAlert"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    [data-testid="stDownloadButton"] button, [data-testid="stDownloadButton"] button * {
        background: linear-gradient(135deg, #DD2476, #FF512F) !important;
        border: none !important; 
        color: white !important; 
        font-weight: 600 !important;
        width: 100%;
    }

    /* HIDE DEFAULT STREAMLIT ELEMENTS */
    header {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    #MainMenu {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    
    /* METRIC CARDS */
    [data-testid="metric-container"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-left: 5px solid #DD2476 !important; 
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important; 
        font-size: 2.2rem !important;
        font-weight: 900 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #38bdf8 !important; 
        font-size: 1rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- NEW: PROFESSIONAL BILINGUAL SENTIMENT ENGINE ---
def analyze_urdu_mood(text):
    text = text.lower()
    score = 0
    neg_words = {
        "bakwas": -3, "ghatiya": -3, "worst": -3, "angry": -3, "fuzool": -2, "bekar": -2, 
        "terrible": -2, "kharab": -1, "masla": -1, "slow": -1, "bad": -1, "issue": -1, 
        "nahi": -1, "kat": -1, "dead": -2, "stuck": -1, "dropping": -1,
        "ganda": -2, "farigh": -2, "farig": -2, "bura": -2, "raddi": -2, "thaka": -1, "masle": -1
    }
    pos_words = {
        "zabardast": 3, "excellent": 3, "best": 3, "behtareen": 2, "acha": 2, 
        "good": 2, "great": 2, "fast": 1, "thanks": 1, "shukriya": 1, "fine": 1
    }
    words = text.split()
    for word in words:
        if word in neg_words:
            score += neg_words[word]
        elif word in pos_words:
            score += pos_words[word]
            
    if score < 0: return "Negative"
    elif score > 0: return "Positive"
    else: return "Neutral"

# --- 6. SESSION STATE ---
if 'df' not in st.session_state: st.session_state['df'] = load_data_fresh()
if 'session_counter' not in st.session_state: st.session_state['session_counter'] = 0

# --- 7. SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=140)
st.sidebar.markdown("---")
user_role = st.sidebar.radio("Select Mode:", ["Customer Portal", "Manager Dashboard"])
st.sidebar.markdown("---")

# --- 8. MAIN INTERFACE ---
if user_role == "Customer Portal":
    st.markdown("<h1>👋 Jazz Intelligent Support</h1>", unsafe_allow_html=True)
    
    if 'customer_logged_in' not in st.session_state: st.session_state['customer_logged_in'] = False
    
    # 🟢 THE FIX 1: SECURE CLOUD AUTHENTICATION LOGIN 🟢
    if not st.session_state['customer_logged_in']:
        with st.form("login"):
            st.markdown("### 🔐 Secure Customer Login")
            st.caption("Please login with your verified Phone Number and Password.")
            phone = st.text_input("Phone Number:", placeholder="e.g. 03001234567")
            password = st.text_input("Password:", type="password", placeholder="Enter your password")
            
            if st.form_submit_button("🔓 Access Portal"):
                if authenticate_customer(phone, password):
                    st.session_state['customer_logged_in'] = True
                    st.session_state['phone'] = phone
                    st.rerun()
                else: 
                    st.error("🚫 Invalid Phone Number or Password. Kripya apne password check karein.")
    else:
        # Chat/User Management
        col1, col2 = st.columns([4,1])
        col1.caption(f"User: {st.session_state['phone']}")
        if col2.button("Log Out"):
            st.session_state['customer_logged_in'] = False
            st.rerun()
        
        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        prompt = st.chat_input("Type complaint...")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    time.sleep(0.5)
                    
                    prediction_probs = model.predict_proba([prompt])[0]
                    max_prob = max(prediction_probs)
                    category = model.classes_[prediction_probs.argmax()]
                    
                    if max_prob < 0.60:
                        category = fallback_intent_classifier(prompt)
                    
                    mood = analyze_urdu_mood(prompt)
                    final_response = get_rag_solution(prompt, category, mood)
                    
                    st.session_state['show_buttons'] = True
                    st.session_state['last_cat'] = category
                    st.session_state['last_mood'] = mood
                    st.session_state['last_txt'] = prompt
                    
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

        if st.session_state.get('show_buttons', False):
            c1, c2 = st.columns(2)
            if c1.button("✅ Solved"):
                row = {'text': st.session_state['last_txt'], 'category': st.session_state['last_cat'], 'status': 'Solved', 'Sentiment': st.session_state['last_mood'], 'Ticket_ID': "Auto", 'Time': (datetime.utcnow() + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"), 'Phone_Number': st.session_state['phone'], 'Data_Source': 'Live'}
                save_ticket(row)
                st.session_state['session_counter'] += 1
                st.success("Saved!")
                st.session_state['show_buttons'] = False
                st.rerun()
            if c2.button("❌ Escalate"): st.session_state['show_form'] = True
        
        if st.session_state.get('show_form', False):
            with st.form("esc"):
                st.write("⚠️ Manager Escalation")
                ph = st.text_input("Confirm Phone:", value=st.session_state.get('phone',''))
                if st.form_submit_button("Send Request"):
                    tid = random.randint(1000,9999)
                    row = {'text': st.session_state['last_txt'], 'category': st.session_state['last_cat'], 'status': 'Escalated', 'Sentiment': st.session_state['last_mood'], 'Ticket_ID': tid, 'Time': (datetime.utcnow() + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"), 'Phone_Number': ph, 'Data_Source': 'Live'}
                    save_ticket(row)
                    st.session_state['session_counter'] += 1
                    st.success(f"Ticket #{tid} Escalated!")
                    st.session_state['show_buttons'] = False
                    st.session_state['show_form'] = False
                    st.rerun()

        st.markdown("---")
        st.markdown("### 🗄️ Recent System Logs")
        recent_df = load_data_fresh().tail(3)
        if not recent_df.empty:
            st.dataframe(recent_df[['Ticket_ID', 'category', 'Sentiment', 'status', 'Time']], use_container_width=True)
        else:
            st.caption("No recent logs found.")

elif user_role == "Manager Dashboard":
    st.sidebar.warning("🔒 Admin Area")
    admin_pass = st.secrets.get("ADMIN_PASSWORD", "admin123")
    if st.sidebar.text_input("Password:", type="password") == admin_pass:
        st.markdown("<h1>📊 Executive Analytics</h1>", unsafe_allow_html=True)
        
        df_full = load_data_fresh()
        today_date_str = (datetime.utcnow() + timedelta(hours=5)).strftime("%Y-%m-%d")
        today_complaints_count = df_full['Time'].astype(str).str.contains(today_date_str).sum()
        
        escalated = len(df_full[df_full['status'] == 'Escalated'])
        solved = len(df_full[df_full['status'].isin(['Solved','Closed'])])
        total = len(df_full)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("New Today", int(today_complaints_count)) 
        c2.metric("Backlog", escalated)
        c3.metric("Solved", solved)
        c4.metric("Total DB", total)
        
        # 🟢 THE FIX 2: VIP CUSTOMER MANAGEMENT BLOCK 🟢
        st.markdown("---")
        st.subheader("👤 Customer Management (Admin Control)")
        with st.container():
            with st.form("add_new_customer"):
                st.markdown("Register a new verified customer to allow them portal access.")
                cc1, cc2 = st.columns(2)
                new_cust_phone = cc1.text_input("New Customer Phone Number:", placeholder="e.g. 03001234567")
                new_cust_pass = cc2.text_input("Assign Password:", type="password", placeholder="e.g. jazz123")
                
                if st.form_submit_button("➕ Register Verified Customer"):
                    if new_cust_phone and new_cust_pass:
                        res = register_new_customer(new_cust_phone, new_cust_pass)
                        if res == "Success":
                            st.success(f"🎉 Customer {new_cust_phone} successfully registered in Cloud DB!")
                        elif res == "Exists":
                            st.warning(f"⚠️ Account for {new_cust_phone} already exists!")
                        else:
                            st.error("🚫 Database Connection Error. Try again.")
                    else:
                        st.error("⚠️ Please fill both Phone Number and Password fields.")

        st.markdown("---")
        st.subheader("🔴 Priority Action Queue")
        
        pending_df = df_full[df_full['status'] == 'Escalated'].copy()
        
        if not pending_df.empty:
            pending_df['Sentiment'] = pending_df['Sentiment'].fillna('Unknown')
            
            def get_mood_priority(mood_text):
                m = str(mood_text).lower()
                if 'negative' in m: return 1
                elif 'neutral' in m: return 2
                elif 'positive' in m: return 3
                else: return 4
            
            pending_df['Priority_Score'] = pending_df['Sentiment'].apply(get_mood_priority)
            pending_df = pending_df.sort_values(by=['Priority_Score', 'Time'], ascending=[True, False])
            
            with st.container(height=450): 
                for index, row in pending_df.iterrows():
                    c1, c2, c3 = st.columns([1.2, 4.5, 1.8]) # ✅ Fixed Resolve Button Size
                    c1.warning(f"#{row.get('Ticket_ID','Old')}")
                    
                    mood_icon = "😡" if row.get('Sentiment') == 'Negative' else ("😊" if row.get('Sentiment') == 'Positive' else "😐")
                    c2.info(f"{mood_icon} **{row['category']}** | 📱 Phone: **{row.get('Phone_Number', 'Not Provided')}** | Mood: {row.get('Sentiment')}\n\n{row['text']}")
                    
                    if c3.button("✅ Resolve", key=f"btn_{index}"):
                        resolve_ticket(row['Ticket_ID'])
                        st.success("Resolved!")
                        time.sleep(0.5)
                        st.rerun()
                    st.markdown("---")
        else:
            st.success("🎉 No escalated complaints in the queue!")
        
        # 🟢 THE FIX 3: ADVANCED FIXED GRAPHS 🟢
        st.markdown("### 📈 Advanced Executive Analytics")
        g1, g2 = st.columns(2)
        with g1:
            v_bar = df_full['category'].value_counts().reset_index()
            v_bar.columns = ['Category', 'Count']
            fig1 = px.bar(v_bar, x='Category', y='Count', title="Complaints by Category", template="plotly_dark", color='Category', text_auto=True)
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#ffffff", size=14), title_font=dict(color="#ffffff", size=20), height=450, margin=dict(t=60, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig1, theme=None, use_container_width=True)
            
        with g2:
            if 'Sentiment' in df_full.columns:
                s = df_full['Sentiment'].value_counts().reset_index()
                s.columns=['Mood','Count']
                fig2 = px.pie(s, names='Mood', values='Count', hole=0.5, title="Mood Radar", color='Mood', color_discrete_map={"Negative":"#FF5252", "Positive":"#69F0AE", "Neutral":"#38bdf8"}, template="plotly_dark")
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#ffffff", size=14), title_font=dict(color="#ffffff", size=20), height=450, margin=dict(t=60, b=20, l=20, r=20), legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
                st.plotly_chart(fig2, theme=None, use_container_width=True)
        
        if not df_full.empty:
            st.markdown("### ☁️ Common Complaint Keywords")
            text = " ".join(title for title in df_full.text.astype(str))
            wc = WordCloud(width=800, height=300, background_color='#0f172a', colormap='plasma').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        csv = df_full.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report", csv, "Jazz_Report.csv", "text/csv")
    else:
        st.error("🚫 Access Denied")