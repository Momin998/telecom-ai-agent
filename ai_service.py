# ai_service.py
import google.generativeai as genai
import streamlit as st

def get_gemini_response(user_text, category, tech_solution, mood):
    """Gemini API ko use kar ke response ko enhance aur translate karta hai."""
    
    # API Key check
    if "GEMINI_API_KEY" not in st.secrets:
        return tech_solution
        
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Professional Enterprise-Grade Prompt Engineering
        prompt = f"""
        You are a highly professional Customer Support Executive for 'Jazz Telecom Pakistan'.
        
        --- TICKET CONTEXT ---
        User Complaint: "{user_text}"
        Issue Category: "{category}"
        User Sentiment: "{mood}"
        OFFICIAL SOLUTION TO DELIVER: "{tech_solution}"
        ----------------------
        
        CRITICAL ENTERPRISE RULES (MUST OBEY):
        1. ZERO HALLUCINATION: You are heavily restricted. You MUST ONLY convey the 'OFFICIAL SOLUTION TO DELIVER'. Do NOT invent your own steps, do NOT promise free balance, MBs, or any compensation.
        2. STRICT LANGUAGE MIRRORING: 
           - If the User Complaint is in Roman Urdu (e.g., 'net slow hai'), your reply MUST be in pure Pakistani Roman Urdu. Strictly avoid Hindi words (like kripya, kshama, prabandh).
           - If the User Complaint is in English, reply in Professional English.
        3. MOOD ADAPTATION (EMPATHY):
           - If Sentiment is "Negative" (angry/frustrated), start by sincerely apologizing for the inconvenience on behalf of Jazz.
           - If Sentiment is "Positive", thank them for their kind words.
           - If Sentiment is "Neutral", be polite and direct.
        4. LENGTH & FORMAT: Keep the response concise (maximum 3-4 lines). Use appropriate telecom emojis (📡, ⚙️, 📱).
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return tech_solution


def fallback_intent_classifier(user_text):
    """Agar ML model fail ho jaye (confidence < 60%), toh Gemini se category pata karta hai."""
    if "GEMINI_API_KEY" not in st.secrets:
        return "Internet" # Default fallback
        
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert Telecom AI router for Jazz Pakistan.
        Classify the following user complaint into exactly ONE of these three categories:
        1. Internet
        2. Billing
        3. Customer Care Call
        
        User Complaint: "{user_text}"
        
        Rules:
        - Return ONLY the exact category name from the list above.
        - Do not add any extra text, punctuation, or explanation.
        """
        
        response = model.generate_content(prompt)
        category = response.text.strip()
        
        valid_categories = ["Internet", "Billing", "Customer Care Call"]
        if category in valid_categories:
            return category
        else:
            return "Internet" # Safe default
    except Exception as e:
        print(f"Fallback Error: {e}")
        return "Internet"