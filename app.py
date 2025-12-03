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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import speech_recognition as sr
import google.generativeai as genai
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Jazz Future AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. GEMINI API SETUP ---
GEMINI_AVAILABLE = False
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_AVAILABLE = True
except:
    pass

def enhance_with_gemini(user_text, category, tech_solution, mood):
    if not GEMINI_AVAILABLE: return tech_solution 
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # --- ROMAN URDU DETECTION PROMPT ---
        prompt = f"""
        Role: You are a polite Customer Support Agent for 'Jazz Telecom Pakistan'.
        
        User Complaint: "{user_text}"
        Category: "{category}"
        User Mood: "{mood}"
        My Technical Solution: "{tech_solution}"
        
        CRITICAL INSTRUCTIONS:
        1. **DETECT LANGUAGE:** If the user writes in **Roman Urdu** (e.g., "Net nai chal raha", "Package lagana hai"), you MUST reply in **Roman Urdu**.
        2. If the user writes in **English**, reply in **English**.
        3. **TONE:** Be empathetic and professional. Use emojis 📡🛠️.
        4. **TASK:** Rewrite 'My Technical Solution' in the detected language. Do not change the meaning. Keep it short (3 lines).
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return tech_solution

# --- 3. DARK HACKER THEME CSS 🎨 ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
        .stApp { background: radial-gradient(circle at 10% 20%, rgb(10, 20, 40) 0%, rgb(5, 10, 25) 90%); color: white; }
        [data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} 
        .kpi-card { background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); }
        .kpi-value { font-size: 36px; font-weight: 700; background: -webkit-linear-gradient(#fff, #ccc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stButton>button { background: linear-gradient(90deg, #D32F2F, #880E4F); color: white; border-radius: 30px; border:none; }
        .stChatMessage { background-color: rgba(255, 255, 255, 0.05); border-radius: 15px; }
        .stTextInput>div>div>input { background-color: rgba(255, 255, 255, 0.1); color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SUPER SMART LOGIC ENGINE (SAARAY SOLUTIONS) 🧠 ---
def get_smart_solution(category, text):
    text = text.lower()
    
    # Safety Guard
    valid_keywords = ["net", "speed", "internet", "wifi", "router", "modem", "slow", "buffer", "lag", "connect", "bill", "balance", "money", "charge", "deduct", "tax", "refund", "payment", "load", "recharge", "sim", "call", "agent", "manager", "staff", "representative", "human", "puk", "block", "offer", "package", "signal", "range", "gaming", "ping", "light", "red", "history", "invoice", "service", "issue", "problem", "masla", "chal", "nahi", "raha", "paise", "kat", "gaye", "batti", "lal"]
    
    if not any(word in text for word in valid_keywords): 
        return None, "🤔 **Out of Scope:** I am a Telecom AI trained only for Internet, Billing, and Sim issues."

    # === INTERNET SOLUTIONS (DETAILED) ===
    if category == "Internet":
        if any(x in text for x in ["red", "blink", "light", "los", "alarm", "lal", "batti"]):
            return "Critical", "🔴 **Hardware Issue:** Check yellow fiber cable behind router."
        elif any(x in text for x in ["slow", "speed", "buffer", "lag", "ahista", "tez"]):
            return "Bandwidth", "📉 **Speed Optimization:** 1. Disconnect extra devices. 2. Restart router (30s off/on)."
        elif any(x in text for x in ["game", "ping", "latency", "pubg", "cod"]):
            return "Gaming", "🎮 **Gaming Issue:** WiFi is unstable for gaming. Use a LAN Cable for 0% loss."
        elif any(x in text for x in ["password", "connect", "access", "login", "change"]):
            return "Access", "🔑 **WiFi Login:** You can reset your WiFi password via the Jazz World App."
        elif any(x in text for x in ["range", "signal", "weak", "kam", "door"]):
            return "Coverage", "📡 **Weak Signal:** 5GHz has short range. Switch to 2.4GHz."
        elif any(x in text for x in ["4g", "lte", "data", "mobile"]):
            return "Mobile Data", "📶 **4G Issue:** Restart phone and check APN settings (jazz.internet)."
        else:
            return "General", "🌐 **Connectivity:** Please restart your router. If issue persists, click 'Escalate'."

    # === BILLING SOLUTIONS (DETAILED) ===
    elif category == "Billing":
        if any(x in text for x in ["tax", "deduction", "cut", "govt", "kat"]):
            return "Tax", "💸 **Tax Info:** 15% Withholding Tax applies on every recharge."
        elif any(x in text for x in ["package", "offer", "subscribe", "laga", "lagana"]):
            return "Subscription", "📦 **Package Status:** Unsubscribe unwanted offers via *111#."
        elif any(x in text for x in ["refund", "balance", "money", "return", "wapis", "double"]):
            return "Refund", "💰 **Refund Claim:** Scanning history... If error found, balance will be reversed."
        elif any(x in text for x in ["history", "bill", "invoice", "check"]):
            return "History", "📅 **Usage History:** View last 6 months history in the App."
        elif any(x in text for x in ["vas", "tune", "song", "game"]):
            return "VAS", "🎵 **Value Added Services:** You are subscribed to VAS. Type 'UNSUB' to 6611."
        else:
            return "General Bill", "💳 **Billing Query:** Check your balance and history in the Jazz World App."

    # === CALL CENTER SOLUTIONS (DETAILED) ===
    elif category == "Customer Care Call":
        if any(x in text for x in ["sim", "block", "puk", "band", "gum", "lock"]):
            return "Security", "🚫 **SIM Security:** Visit Jazz Franchise with CNIC for Biometric verification."
        elif any(x in text for x in ["mnp", "port", "switch", "network"]):
            return "MNP", "📲 **Port In:** Visit Franchise to switch to Jazz network."
        elif any(x in text for x in ["ownership", "transfer", "name"]):
            return "Ownership", "📝 **Transfer:** Both parties must visit Franchise for biometric."
        else:
            return "Human Agent", "🎧 **Support:** Our lines are busy. Connecting to a human agent shortly."
    
    return "General", "👉 Request forwarded to Technical Team."

# --- 5. DATA LOAD (SAFE & ROBUST) ---
def load_data_fresh():
    try:
        df = pd.read_csv('Comcast.csv')
        if 'Customer Complaint' in df.columns:
            df['Customer Complaint'] = df['Customer Complaint'].str.replace('Comcast', 'Jazz', case=False)
            df = df.rename(columns={'Customer Complaint': 'text', 'Received Via': 'category', 'Status': 'status'})
            df = df.dropna(subset=['text', 'category'])
        
        # Backlog Logic
        df['status'] = df['status'].replace({'Open': 'Escalated', 'Pending': 'Escalated'})
        
        for col in ['Sentiment', 'Ticket_ID', 'Time', 'Phone_Number', 'Data_Source']:
            if col not in df.columns: df[col] = "N/A"
            
        df['Date_Parsed'] = pd.to_datetime(df['Time'], errors='coerce').dt.date
        return df
    except: 
        return pd.DataFrame(columns=['text', 'category', 'status', 'Sentiment', 'Ticket_ID', 'Time', 'Phone_Number', 'Data_Source'])

if 'df' not in st.session_state: st.session_state['df'] = load_data_fresh()

# Session Counter
if 'session_counter' not in st.session_state: st.session_state['session_counter'] = 0

# --- 6. TRAINING ---
big_training_data = [{"text": "net slow", "category": "Internet", "status": "Escalated"}, {"text": "bill wrong", "category": "Billing", "status": "Escalated"}, {"text": "call agent", "category": "Customer Care Call", "status": "Escalated"}]
df_extra = pd.DataFrame(big_training_data * 50)
df_train = pd.concat([st.session_state['df'], df_extra], ignore_index=True)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df_train['text'], df_train['category'])

# --- 7. SIDEBAR (PROFESSIONAL LOGO) ---
# New 3D Robot Icon (Professional)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=140)
# Gemini Status Hidden (As requested)
st.sidebar.markdown("---")
user_role = st.sidebar.radio("Select Mode:", ["Customer Portal", "Manager Dashboard"])
st.sidebar.markdown("---")

# --- 8. MAIN LOGIC ---
if user_role == "Customer Portal":
    st.markdown("<h1 style='text-align: center; background: -webkit-linear-gradient(#FF512F, #DD2476); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 40px;'>👋 Jazz Intelligent Support</h1>", unsafe_allow_html=True)
    
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
        col1, col2 = st.columns([4,1])
        col1.caption(f"User: {st.session_state['phone']}")
        if col2.button("Log Out"):
            st.session_state['customer_logged_in'] = False
            st.rerun()
        
        # Voice
        col_mic, _ = st.columns([1, 5])
        voice_text = None
        with col_mic:
            if st.button("🎙️ Speak"):
                r = sr.Recognizer()
                with st.spinner("Listening..."):
                    try:
                        with sr.Microphone() as source:
                            audio = r.listen(source, timeout=5)
                            voice_text = r.recognize_google(audio)
                            st.success(f"Voice: {voice_text}")
                    except: st.error("Audio Error")

        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        prompt = st.chat_input("Type complaint...")
        if voice_text: prompt = voice_text

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
                        
                        # GEMINI CALL (Roman Urdu Supported)
                        final_response = enhance_with_gemini(prompt, category, raw_solution, mood)
                        
                        st.session_state['show_buttons'] = True
                        st.session_state['last_cat'] = category
                        st.session_state['last_mood'] = mood
                        st.session_state['last_txt'] = prompt
                    
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

        if st.session_state.get('show_buttons', False):
            c1, c2 = st.columns(2)
            if c1.button("✅ Solved"):
                row = {'text': st.session_state['last_txt'], 'category': st.session_state['last_cat'], 'status': 'Solved', 'Sentiment': st.session_state['last_mood'], 'Ticket_ID': "Auto", 'Time': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Phone_Number': st.session_state['phone'], 'Data_Source': 'Live'}
                current_df = load_data_fresh()
                updated_df = pd.concat([current_df, pd.DataFrame([row])], ignore_index=True)
                updated_df.to_csv('Comcast.csv', index=False)
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
                    row = {'text': st.session_state['last_txt'], 'category': st.session_state['last_cat'], 'status': 'Escalated', 'Sentiment': st.session_state['last_mood'], 'Ticket_ID': tid, 'Time': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Phone_Number': ph, 'Data_Source': 'Live'}
                    current_df = load_data_fresh()
                    updated_df = pd.concat([current_df, pd.DataFrame([row])], ignore_index=True)
                    updated_df.to_csv('Comcast.csv', index=False)
                    st.session_state['session_counter'] += 1
                    st.success(f"Ticket #{tid} Escalated!")
                    st.session_state['show_buttons'] = False
                    st.session_state['show_form'] = False
                    st.rerun()

elif user_role == "Manager Dashboard":
    st.sidebar.warning("🔒 Admin Area")
    if st.sidebar.text_input("Password:", type="password") == "admin123":
        st.markdown("""<h2 style='background: -webkit-linear-gradient(#FF512F, #DD2476); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;'>📊 Executive Analytics</h2>""", unsafe_allow_html=True)
        
        col_search, col_refresh = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Global Search (Ticket, Phone, Name)...", placeholder="Search...")
        with col_refresh:
            if st.button("🔄 Refresh Feed"): st.rerun()

        df_full = load_data_fresh()
        
        # Date Filter
        st.sidebar.markdown("### 📅 Time Filter")
        filter_mode = st.sidebar.radio("View Data:", ["All Time", "Specific Date"])
        selected_date = None
        if filter_mode == "Specific Date":
            selected_date = st.sidebar.date_input("Select Date:", datetime.now())
            if 'Date_Parsed' in df_full.columns:
                df_view = df_full[df_full['Date_Parsed'] == selected_date]
            else: df_view = df_full
        else:
            df_view = df_full

        if search_term:
            df_view = df_view[df_view.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
            st.info(f"Search Results: {len(df_view)}")

        # KPIs
        new_today = st.session_state['session_counter']
        escalated = len(df_full[df_full['status'] == 'Escalated'])
        solved = len(df_view[df_view['status'].isin(['Solved','Closed'])])
        total = len(df_view)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:#FF5252;'>{new_today}</div><div class='kpi-label'>New Today</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:#FFAB40;'>{escalated}</div><div class='kpi-label'>Backlog</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:#69F0AE;'>{solved}</div><div class='kpi-label'>Solved</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total}</div><div class='kpi-label'>Total DB</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action Table
        st.subheader("🔴 Priority Action Queue")
        pending_df = df_view if search_term else df_full[df_full['status'] == 'Escalated'].copy()
        
        if not pending_df.empty:
            priority_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            pending_df['priority'] = pending_df['Sentiment'].map(priority_map).fillna(1)
            pending_df = pending_df.sort_values(by=['priority', 'Time'], ascending=[True, False])
            
            for index, row in pending_df.head(10).iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([1, 4, 1])
                    mood_icon = "😡" if row['Sentiment'] == 'Negative' else "😐"
                    c1.warning(f"#{row.get('Ticket_ID','Old')}")
                    c2.info(f"{mood_icon} **{row['category']}**: {row['text']}\n🕒 {row.get('Time','N/A')} | 📞 {row.get('Phone_Number','N/A')}")
                    
                    if c3.button("✅ Resolve", key=f"btn_{index}"):
                        full_df = pd.read_csv('Comcast.csv')
                        text_col = 'text' if 'text' in full_df.columns else 'Customer Complaint'
                        status_col = 'status' if 'status' in full_df.columns else 'Status'
                        mask = (full_df[text_col] == row['text'])
                        if mask.any():
                            idx = full_df[mask].index[-1]
                            full_df.at[idx, status_col] = 'Solved'
                            full_df.to_csv('Comcast.csv', index=False)
                            st.success("Resolved!")
                            time.sleep(0.5)
                            st.rerun()
                    st.markdown("---")
        else:
            st.success("🎉 Inbox Zero!")

        with st.expander("📂 View Analytics Graphs"):
             g1, g2 = st.columns(2)
             with g1:
                 v = df_view['category'].value_counts().reset_index()
                 v.columns=['Category','Count']
                 st.plotly_chart(px.pie(v, names='Category', values='Count', title="Category Distribution", template="plotly_dark"), use_container_width=True)
             with g2:
                 if 'Sentiment' in df_view.columns:
                    s = df_view['Sentiment'].value_counts().reset_index()
                    s.columns=['Mood','Count']
                    st.plotly_chart(px.pie(s, names='Mood', values='Count', hole=0.5, title="Mood Radar", color='Mood', color_discrete_map={"Negative":"#FF5252", "Positive":"#69F0AE", "Neutral":"#448AFF"}, template="plotly_dark"), use_container_width=True)
             
             if not df_view.empty:
                st.subheader("☁️ Keyword Cloud")
                text = " ".join(title for title in df_view.text.astype(str))
                wc = WordCloud(width=800, height=300, background_color='#0e1117', colormap='Reds').generate(text)
                fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

             csv = df_view.to_csv(index=False).encode('utf-8')
             st.download_button("📥 Download Report", csv, "Jazz_Report.csv", "text/csv")

    else:
        st.error("🚫 Access Denied")