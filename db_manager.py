# db_manager.py
import pandas as pd
import streamlit as st
from supabase import create_client, Client

# --- SUPABASE CONNECTION SETUP ---
@st.cache_resource
def init_connection():
    """Supabase API se connect karne ka secure tareeqa"""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

supabase = init_connection()

def load_data_fresh():
    """Cloud Database (Supabase) se tickets load karta hai."""
    try:
        # Supabase se 'tickets' table ka saara data fetch karna
        response = supabase.table("tickets").select("*").execute()
        data = response.data
        
        # Agar table khali hai ya abhi tak data nahi aaya
        if not data:
            df = pd.DataFrame(columns=['text', 'category', 'status', 'Sentiment', 'Ticket_ID', 'Time', 'Phone_Number', 'Data_Source'])
        else:
            df = pd.DataFrame(data)
        
        # ---> PURANI LOGIC (100% INTACT) <---
        # Status aur Date ki safai
        df['status'] = df['status'].replace({'Open': 'Escalated', 'Pending': 'Escalated'})
        if 'Time' in df.columns:
            df['Date_Parsed'] = pd.to_datetime(df['Time'], errors='coerce').dt.date
        return df
        
    except Exception as e: 
        print(f"Database Load Error: {e}")
        return pd.DataFrame(columns=['text', 'category', 'status', 'Sentiment', 'Ticket_ID', 'Time', 'Phone_Number', 'Data_Source'])

def save_ticket(row_dict):
    """Naya ticket real-time Cloud DB mein save karta hai."""
    try:
        # Supabase mein directly dictionary insert karna
        supabase.table("tickets").insert(row_dict).execute()
        return True
    except Exception as e:
        print(f"Database Save Error: {e}")
        return False

def resolve_ticket(ticket_text):
    """Manager Dashboard se ticket ko Supabase mein Solved mark karta hai."""
    try:
        # Jo text match kare, uska status 'Solved' update kar do
        supabase.table("tickets").update({"status": "Solved"}).eq("text", ticket_text).execute()
        return True
    except Exception as e:
        print(f"Database Update Error: {e}")
        return False