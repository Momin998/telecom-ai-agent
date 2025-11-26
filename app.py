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
        prompt = f"""
        Act as a polite Customer Support Agent for 'Jazz Telecom'.
        User Complaint: "{user_text}"
        Category: "{category}"
        User Mood: "{mood}"
        Technical Solution: "{tech_solution}"
        
        Task: Rewrite the Technical Solution in a natural, helpful, and professional tone.
        - Keep it short (3 lines max).
        - If mood is negative, show empathy.
        - Use emojis 📡.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return tech_solution

# --- 3. DARK HACKER THEME CSS (WITH GRADIENT HEADER) 🎨 ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
        
        /* Deep Dark Background */
        .stApp { 
            background: radial-gradient(circle at 10% 20%, rgb(10, 20, 40) 0%, rgb(5, 10, 25) 90%); 
            color: white; 
        }
        
        /* Glass Sidebar */
        [data-testid="stSidebar"] { 
            background-color: rgba(255, 255, 255, 0.05); 
            backdrop-filter: blur(10px); 
            border-right: 1px solid rgba(255, 255, 255, 0.1); 
        }
        
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;} 
        
        /* Glowing Cards */
        .kpi-card { 
            background: rgba(255, 255, 255, 0.05); 
            border-radius: 16px; 
            padding: 20px; 
            text-align: center; 
            backdrop-filter: blur(10px); 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); 
            transition: transform 0.3s ease; 
        }
        .kpi-card:hover { 
            transform: translateY(-5px); 
            border-color: #D32F2F; 
            box-shadow: 0 0 20px rgba(211, 47, 47, 0.4); 
        }
        .kpi-value { 
            font-size: 36px; 
            font-weight: 700; 
            background: -webkit-linear-gradient(#fff, #ccc); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        .kpi-label { 
            font-size: 12px; color: #aaa; letter-spacing: 1px; text-transform: uppercase; 
        }
        
        /* Neon Buttons */
        .stButton>button { 
            background: linear-gradient(90deg, #D32F2F, #880E4F); 
            color: white; border: none; border-radius: 30px; 
            padding: 12px 30px; font-weight: 600; transition: 0.4s; width: 100%; 
        }
        .stButton>button:hover { 
            background: linear-gradient(90deg, #FF5252, #D32F2F); 
            transform: scale(1.02); 
            box-shadow: 0 0 15px rgba(255, 82, 82, 0.6); 
        }
        
        /* Chat & Inputs */
        .stChatMessage { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 15px; }
        .stTextInput>div>div>input { background-color: rgba(255, 255, 255, 0.1); color: white; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); }
    </style>
""", unsafe_allow_html=True)

# --- 4. SMART LOGIC ENGINE ---
def get_smart_solution(category, text):
    text = text.lower()
    valid_keywords = ["net", "speed", "internet", "wifi", "router", "modem", "slow", "buffer", "lag", "connect", "bill", "balance", "money", "charge", "deduct", "tax", "refund", "payment", "load", "recharge", "sim", "call", "agent", "manager", "staff", "representative", "human", "puk", "block", "offer", "package", "signal", "range", "gaming", "ping", "light", "red", "history", "invoice", "service", "issue", "problem"]
    if not any(word in text for word in valid_keywords): return None, "🤔 **Out of Scope:** I am trained only for Telecom issues."

    if category == "Internet":
        if any(x in text for x in ["red", "blink", "light"]): return "Critical", "🔴 **Hardware:** Check yellow fiber cable."
        elif any(x in text for x in ["slow", "speed", "buffer"]): return "Bandwidth", "📉 **Speed:** Restart router (30s off/on)."
        elif any(x in text for x in ["game", "ping"]): return "Gaming", "🎮 **Gaming:** Use LAN Cable."
        elif any(x in text for x in ["password", "connect"]): return "Access", "🔑 **Login:** Reset WiFi password via App."
        else: return "General", "🌐 **Connectivity:** Restart router."
    elif category == "Billing":
        if any(x in text for x in ["tax", "deduction"]): return "Tax", "💸 **Tax:** 15% Tax applies."
        elif any(x in text for x in ["package", "offer"]): return "Subscription", "📦 **Offer:** Unsubscribe via *111#."
        elif any(x in text for x in ["refund", "balance", "money"]): return "Refund", "💰 **Refund:** Checking history... Error balance reversed."
        else: return "General Bill", "💳 **Billing:** Check usage history in App."
    elif category == "Customer Care Call":
        if any(x in text for x in ["sim", "block", "puk"]): return "Security", "🚫 **SIM:** Visit Franchise for Biometric."
        else: return "Human Agent", "🎧 **Support:** Connecting to agent..."
    return "General", "👉 Request forwarded to Tech Team."

# --- 5. DATA LOAD ---
@st.cache_data
def load_initial_data():
    try:
        df = pd.read_csv('Comcast.csv')
        if 'Customer Complaint' in df.columns:
            df['Customer Complaint'] = df['Customer Complaint'].str.replace('Comcast', 'Jazz', case=False)
            df['Customer Complaint'] = df['Customer Complaint'].str.replace('comcast', 'Jazz', case=False)
            df = df.rename(columns={'Customer Complaint': 'text', 'Received Via': 'category', 'Status': 'status'})
            df = df.dropna(subset=['text', 'category'])
        if 'Sentiment' not in df.columns: df['Sentiment'] = 'Neutral'
        if 'Ticket_ID' not in df.columns: df['Ticket_ID'] = "N/A"
        if 'Time' not in df.columns: df['Time'] = "N/A"
        if 'Phone_Number' not in df.columns: df['Phone_Number'] = "N/A"
        return df
    except: return pd.DataFrame(columns=['text', 'category', 'status', 'Sentiment', 'Ticket_ID', 'Time', 'Phone_Number'])

if 'df' not in st.session_state: st.session_state['df'] = load_initial_data()
df = st.session_state['df']

# --- 6. BIG DATA TRAINING ---
big_training_data = [
    {"text": "overcharged month", "category": "Billing", "status": "Open"},
    {"text": "balance deducted", "category": "Billing", "status": "Open"},
    {"text": "refund money", "category": "Billing", "status": "Open"},
    {"text": "talk to human agent", "category": "Customer Care Call", "status": "Pending"},
    {"text": "internet speed slow", "category": "Internet", "status": "Open"},
    {"text": "router red light", "category": "Internet", "status": "Open"},
    {"text": "sim blocked", "category": "Customer Care Call", "status": "Open"},
    {"text": "pubg ping high", "category": "Internet", "status": "Open"},
    {"text": "tax deduction", "category": "Billing", "status": "Solved"},
    {"text": "check bill history", "category": "Billing", "status": "Open"},
]
df_extra = pd.DataFrame(big_training_data * 50)
df_train = pd.concat([df, df_extra], ignore_index=True)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df_train['text'], df_train['category'])

# --- 7. SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Jazz_logo.png/320px-Jazz_logo.png", width=140)
st.sidebar.markdown("### 🧠 AI Status")
if GEMINI_AVAILABLE: st.sidebar.success("✅ Gemini 2.0: Active")
else: st.sidebar.warning("⚠️ Gemini: Inactive")
st.sidebar.markdown("---")
user_role = st.sidebar.radio("Select Mode:", ["Customer Portal", "Manager Dashboard"])
st.sidebar.markdown("---")

# --- 8. MAIN LOGIC ---
if user_role == "Customer Portal":
    # GRADIENT HEADER (ADVANCED LOOK)
    st.markdown("""
        <h1 style='text-align: center; background: -webkit-linear-gradient(#FF512F, #DD2476); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 40px;'>
        👋 Jazz Intelligent Support
        </h1>
        <p style='text-align: center; color: #aaa;'>Next-Gen AI Powered Assistance</p>
    """, unsafe_allow_html=True)
    
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
        
        # Voice Input
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
                with st.spinner("AI Thinking..."):
                    time.sleep(0.5)
                    prompt_lower = prompt.lower()
                    greetings = ["hello", "hi", "salam"]
                    thanks = ["thank", "ok", "solved"]
                    
                    if any(x in prompt_lower for x in greetings):
                        response = "👋 **Hello!** I am your Jazz AI Assistant. How can I help you?"
                    elif any(x in prompt_lower for x in thanks):
                        response = "🌟 **My Pleasure!**"
                        st.balloons()
                    else:
                        category = model.predict([prompt])[0]
                        sol_type, raw_solution = get_smart_solution(category, prompt)
                        
                        if sol_type is None:
                            response = raw_solution
                            st.session_state['show_buttons'] = False
                        else:
                            blob = TextBlob(prompt)
                            mood = "Negative" if blob.sentiment.polarity < -0.1 else "Neutral"
                            final_response = enhance_with_gemini(prompt, category, raw_solution, mood)
                            st.session_state['show_buttons'] = True
                            st.session_state['last_cat'] = category
                            st.session_state['last_mood'] = mood
                            st.session_state['last_txt'] = prompt
                            response = final_response
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.get('show_buttons', False):
            c1, c2 = st.columns(2)
            if c1.button("✅ Solved"):
                row = {'text': st.session_state['last_txt'], 'category': st.session_state['last_cat'], 'status': 'Solved', 'Sentiment': st.session_state['last_mood'], 'Ticket_ID': "Auto", 'Time': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Phone_Number': st.session_state['phone']}
                st.session_state['df'] = pd.concat([st.session_state['df'], pd.DataFrame([row])], ignore_index=True)
                st.session_state['df'].to_csv('Comcast.csv', index=False)
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
                    row = {'text': st.session_state['last_txt'], 'category': st.session_state['last_cat'], 'status': 'Escalated', 'Sentiment': st.session_state['last_mood'], 'Ticket_ID': tid, 'Time': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Phone_Number': ph}
                    st.session_state['df'] = pd.concat([st.session_state['df'], pd.DataFrame([row])], ignore_index=True)
                    st.session_state['df'].to_csv('Comcast.csv', index=False)
                    st.success(f"Ticket #{tid} Escalated!")
                    st.session_state['show_buttons'] = False
                    st.session_state['show_form'] = False
                    st.rerun()

elif user_role == "Manager Dashboard":
    st.sidebar.warning("🔒 Admin Area")
    if st.sidebar.text_input("Password:", type="password") == "admin123":
        st.markdown("""
            <h2 style='background: -webkit-linear-gradient(#FF512F, #DD2476); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;'>
            📊 Executive Analytics
            </h2>""", unsafe_allow_html=True)
        
        filter_stat = st.sidebar.multiselect("Filter:", options=df['status'].unique(), default=df['status'].unique())
        df_view = df[df['status'].isin(filter_stat)]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(df_view)}</div><div class='kpi-label'>Total</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:#FF5252;'>{len(df_view[df_view['category'].str.contains('Internet')])}</div><div class='kpi-label'>Net</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:#69F0AE;'>{len(df_view[df_view['status'].isin(['Solved','Closed'])])}</div><div class='kpi-label'>Solved</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:#FFAB40;'>{len(df_view[df_view['status']=='Escalated'])}</div><div class='kpi-label'>Escalated</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            v = df_view['category'].value_counts().reset_index()
            v.columns=['Category','Count']
            st.plotly_chart(px.bar(v, x='Category', y='Count', color='Category', template="plotly_dark"), use_container_width=True)
        with g2:
            if 'Sentiment' in df_view.columns:
                s = df_view['Sentiment'].value_counts().reset_index()
                s.columns=['Mood','Count']
                st.plotly_chart(px.pie(s, names='Mood', values='Count', color='Mood', color_discrete_map={"Negative":"#FF5252", "Positive":"#69F0AE", "Neutral":"#448AFF"}, template="plotly_dark"), use_container_width=True)
        
        st.markdown("---")
        if not df_view.empty:
            st.subheader("☁️ Keyword Cloud")
            text = " ".join(title for title in df_view.text.astype(str))
            wc = WordCloud(width=800, height=300, background_color='#0e1117', colormap='Reds').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
        csv = df_view.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "Jazz_Report.csv", "text/csv")
        st.dataframe(df.tail(10))
    else:
        st.error("🚫 Access Denied")