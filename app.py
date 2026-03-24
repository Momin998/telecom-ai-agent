# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
from datetime import datetime, timedelta
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# ---> 1. MODULES IMPORT (100% Intact Backend)
from db_manager import load_data_fresh, save_ticket, resolve_ticket 
from ml_engine import train_bilingual_model
from ai_service import get_gemini_response 

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
        caret-color: #38bdf8 !important; /* Sidebar blue blinking cursor */
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
        border: 1px solid #475569 !important; /* Visible default border */
        border-radius: 10px !important;
        color: white !important;
        padding: 10px !important;
        caret-color: #DD2476 !important; /* Jazz Pink Blinking Cursor */
    }
    
    /* Placeholders clearly visible */
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
    /* 🔴 THE FIX: CHAT TEXT BRIGHT WHITE & BOLD 🔴 */
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
    
    /* 🔴 THE FIX: READABLE COMPLAINTS & TICKET BOXES (FOR BOTH THEMES) 🔴 */
    [data-testid="stAlert"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* 🔴 THE FIX: DOWNLOAD BUTTON BACKGROUND AND TEXT 🔴 */
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
    
    /* 🔴 THE FIX: METRIC CARDS (EXECUTIVE ANALYTICS) 🔴 */
    [data-testid="metric-container"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-left: 5px solid #DD2476 !important; /* Pink Accent Line */
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important; /* Bright White Numbers */
        font-size: 2.2rem !important;
        font-weight: 900 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #38bdf8 !important; /* Cyber Blue Labels */
        font-size: 1rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 5. LOGIC ENGINE (If-Else) ---
def get_smart_solution(category, text):
    text = text.lower()
    valid_keywords = ["net", "speed", "internet", "wifi", "router", "modem", "slow", "buffer", "lag", "connect", "bill", "balance", "money", "charge", "deduct", "tax", "refund", "payment", "load", "recharge", "sim", "call", "agent", "manager", "staff", "representative", "human", "puk", "block", "offer", "package", "signal", "range", "gaming", "ping", "light", "red", "history", "invoice", "service", "issue", "problem", "masla", "chal", "nahi", "raha", "paise", "kat", "gaye", "batti", "lal", "ahista", "aista", "dropping", "disconnects", "dead", "break", "working", "stuck", "rha", "automatic", "bina", "incorrect", "extra", "double", "zyada", "wapis", "karo", "vas", "tune", "song", "laga", "error", "check", "unwanted", "account", "meri", "band", "lock", "need", "talk", "center", "apne", "naam", "karni", "ownership", "transfer", "biometric", "mnp", "port", "network", "change", "dusri", "franchise", "location", "address", "batao", "nearest", "care", "gum", "duplicate", "nikalni", "escalate", "complaint", "khatam", "charges", "bundle", "detail", "thumb", "lost", "kahan"]
    
    if not any(word in text for word in valid_keywords): 
        return None, "🤔 **Out of Scope:** I am a Telecom AI trained only for Internet, Billing, and Sim issues."

    if category == "Internet":
        if any(x in text for x in ["red", "blink", "light", "los", "alarm", "lal", "batti"]): return "Critical", "🔴 **Hardware Issue:** Check yellow fiber cable behind router."
        elif any(x in text for x in ["slow", "speed", "buffer", "lag", "ahista", "aista", "tez", "stuck", "dead", "crawling"]): return "Bandwidth", "📉 **Speed Optimization:** 1. Disconnect extra devices. 2. Restart router (30s off/on)."
        elif any(x in text for x in ["game", "ping", "latency", "pubg", "cod"]): return "Gaming", "🎮 **Gaming Issue:** WiFi is unstable for gaming. Use a LAN Cable for 0% loss."
        elif any(x in text for x in ["password", "connect", "access", "login", "change", "working"]): return "Access", "🔑 **WiFi Login:** You can reset your WiFi password via the Jazz World App."
        elif any(x in text for x in ["range", "signal", "weak", "kam", "door", "dropping", "disconnects", "drop"]): return "Coverage", "📡 **Weak Signal:** 5GHz has short range. Switch to 2.4GHz."
        elif any(x in text for x in ["4g", "lte", "data", "mobile", "3g"]): return "Mobile Data", "📶 **4G Issue:** Restart phone and check APN settings (jazz.internet)."
        else: return "General", "🌐 **Connectivity:** Please restart your router."

    elif category == "Billing":
        if any(x in text for x in ["tax", "deduction", "cut", "govt", "kat", "automatic", "extra", "charges"]): return "Tax", "💸 **Tax Info:** 15% Withholding Tax applies on every recharge."
        elif any(x in text for x in ["package", "offer", "subscribe", "laga", "lagana", "unwanted", "bundle"]): return "Subscription", "📦 **Package Status:** Unsubscribe unwanted offers via *111#."
        elif any(x in text for x in ["refund", "balance", "money", "return", "wapis", "double", "incorrect", "wrong"]): return "Refund", "💰 **Refund Claim:** Scanning history... If error found, balance will be reversed."
        elif any(x in text for x in ["history", "bill", "invoice", "check", "account", "detail"]): return "History", "📅 **Usage History:** View last 6 months history in the App."
        elif any(x in text for x in ["vas", "tune", "song", "game", "caller"]): return "VAS", "🎵 **Value Added Services:** You are subscribed to VAS. Type 'UNSUB' to 6611."
        else: return "General Bill", "💳 **Billing Query:** Check your balance and history in the Jazz World App."

    elif category == "Customer Care Call":
        if any(x in text for x in ["sim", "block", "puk", "band", "gum", "lock", "duplicate", "nikalni", "lost"]): return "Security", "🚫 **SIM Security:** Visit Jazz Franchise with CNIC for Biometric verification."
        elif any(x in text for x in ["mnp", "port", "switch", "network", "dusri", "change"]): return "MNP", "📲 **Port In:** Visit Franchise to switch to Jazz network."
        elif any(x in text for x in ["ownership", "transfer", "name", "naam", "apne", "biometric", "thumb"]): return "Ownership", "📝 **Transfer:** Both parties must visit Franchise for biometric."
        elif any(x in text for x in ["franchise", "location", "address", "batao", "nearest", "kahan"]): return "Location", "📍 **Franchise Locator:** Find nearest franchise on Google Maps."
        else: return "Human Agent", "🎧 **Support:** Connecting to a human agent shortly."
    
    return "General", "👉 Request forwarded to Technical Team."

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
    if not st.session_state['customer_logged_in']:
        with st.form("login"):
            phone = st.text_input("Phone Number:", placeholder="0300xxxxxxx")
            if st.form_submit_button("🔓 Access"):
                if len(phone) == 11:
                    st.session_state['customer_logged_in'] = True
                    st.session_state['phone'] = phone
                    st.rerun()
                else: st.error("Invalid Number")
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
                    category = model.predict([prompt])[0]
                    sol_type, raw_solution = get_smart_solution(category, prompt)
                    
                    if sol_type is None:
                        final_response = raw_solution
                        st.session_state['show_buttons'] = False
                    else:
                        blob = TextBlob(prompt)
                        mood = "Negative" if blob.sentiment.polarity < -0.1 else "Neutral"
                        # AI Service Call
                        final_response = get_gemini_response(prompt, category, raw_solution, mood)
                        
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
        
        # 🟢 THE FIX 1: "New Today" REAL-TIME DATABASE COUNTER (PKT SYNCED) 🟢
        today_date_str = (datetime.utcnow() + timedelta(hours=5)).strftime("%Y-%m-%d")
        # Ensure Time column is string to avoid errors, then check for today's date
        today_complaints_count = df_full['Time'].astype(str).str.contains(today_date_str).sum()
        
        escalated = len(df_full[df_full['status'] == 'Escalated'])
        solved = len(df_full[df_full['status'].isin(['Solved','Closed'])])
        total = len(df_full)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("New Today", int(today_complaints_count)) 
        c2.metric("Backlog", escalated)
        c3.metric("Solved", solved)
        c4.metric("Total DB", total)
        
        st.markdown("---")
        st.subheader("🔴 Priority Action Queue")
        
        # 🟢 THE FIX 2 & 3: SMART SORTING & SCROLLABLE CONTAINER 🟢
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
                    c1, c2, c3 = st.columns([1, 4, 1])
                    c1.warning(f"#{row.get('Ticket_ID','Old')}")
                    
                    mood_icon = "😡" if row.get('Sentiment') == 'Negative' else ("😊" if row.get('Sentiment') == 'Positive' else "😐")
                    c2.info(f"{mood_icon} **{row['category']}** | Mood: {row.get('Sentiment')}\n\n{row['text']}")
                    
                    if c3.button("✅ Resolve", key=f"btn_{index}"):
                        resolve_ticket(row['text'])
                        st.success("Resolved!")
                        time.sleep(0.5)
                        st.rerun()
                    st.markdown("---")
        else:
            st.success("🎉 No escalated complaints in the queue!")
        
        # GRAPHS SECTION (Intact)
        st.markdown("### 📈 Advanced Executive Analytics")
        g1, g2 = st.columns(2)
        with g1:
            v = df_full['category'].value_counts().reset_index()
            v.columns=['Category','Count']
            fig1 = px.pie(v, names='Category', values='Count', title="Category Distribution", template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
            
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff", size=14),
                title_font=dict(color="#ffffff", size=20)
            )
            st.plotly_chart(fig1, theme=None, use_container_width=True)
            
        with g2:
            if 'Sentiment' in df_full.columns:
                s = df_full['Sentiment'].value_counts().reset_index()
                s.columns=['Mood','Count']
                fig2 = px.pie(s, names='Mood', values='Count', hole=0.5, title="Mood Radar", color='Mood', color_discrete_map={"Negative":"#FF5252", "Positive":"#69F0AE", "Neutral":"#38bdf8"}, template="plotly_dark")
                
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#ffffff", size=14),
                    title_font=dict(color="#ffffff", size=20)
                )
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