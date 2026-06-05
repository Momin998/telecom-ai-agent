# db_manager.py
import pandas as pd
import streamlit as st
from supabase import create_client

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
        response = supabase.table("tickets").select("*").execute()
        data = response.data
        
        if not data:
            return pd.DataFrame(columns=['text', 'category', 'status', 'Sentiment', 'Ticket_ID', 'Time', 'Phone_Number', 'Data_Source'])
            
        df = pd.DataFrame(data)
        
        # 🔥 THE FIX: Supabase ke small letters ko wapis App ke Capital letters mein badalna
        df = df.rename(columns={
            "sentiment": "Sentiment",
            "ticket_id": "Ticket_ID",
            "time": "Time",
            "phone_number": "Phone_Number",
            "data_source": "Data_Source"
        })
        
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
        # 🔥 THE FIX: App ke Capital letters ko Supabase ke small letters mein convert kar ke bhejna
        db_row = {
            "text": row_dict.get("text"),
            "category": row_dict.get("category"),
            "status": row_dict.get("status"),
            "sentiment": row_dict.get("Sentiment"),
            "ticket_id": str(row_dict.get("Ticket_ID")), # Convert to string to avoid mismatch
            "time": row_dict.get("Time"),
            "phone_number": row_dict.get("Phone_Number"),
            "data_source": row_dict.get("Data_Source"),
            "ai_response": row_dict.get("ai_response")
        }
        
        supabase.table("tickets").insert(db_row).execute()
        return True
    except Exception as e:
        print(f"Database Save Error: {e}")
        return False
def resolve_ticket(ticket_id):
    """Manager Dashboard se ticket ko uske Unique ID se Supabase mein Solved mark karta hai."""
    try:
        # 🔥 THE FIX: 'text' ki bajaye 'ticket_id' column use kiya, aur usay string mein bheja
        supabase.table("tickets").update({"status": "Solved"}).eq("ticket_id", str(ticket_id)).execute()
        return True
    except Exception as e:
        print(f"Database Update Error: {e}")
        return False
    
    # --- NAYE AUTHENTICATION FUNCTIONS (Add at the very bottom of db_manager.py) ---

def authenticate_customer(phone, password):
    """Checks if the phone and password match the Supabase customers table."""
    try:
        response = supabase.table('customers').select('*').eq('phone_number', phone).eq('password', password).execute()
        # Agar data list mein koi record aya hai, matlab user original hai
        if len(response.data) > 0:
            return True
        return False
    except Exception as e:
        print(f"Auth Error: {e}")
        return False

def register_new_customer(phone, password):
    """Admin function to register a new customer in Supabase."""
    try:
        # Pehle check karein ke is number ka account pehle se toh nahi bana hua
        check = supabase.table('customers').select('*').eq('phone_number', phone).execute()
        if len(check.data) > 0:
            return "Exists"
        
        # Agar naya user hai toh database mein save kar do
        data = supabase.table('customers').insert({"phone_number": phone, "password": password}).execute()
        return "Success"
    except Exception as e:
        print(f"Register Error: {e}")
        return "Error"