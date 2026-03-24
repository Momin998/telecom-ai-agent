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
        
        # Professional Prompt Engineering
        prompt = f"""
        Role: You are a polite Customer Support Agent for 'Jazz Telecom Pakistan'.
        
        User Complaint: "{user_text}"
        Category: "{category}"
        User Mood: "{mood}"
        My Technical Solution: "{tech_solution}"
        
        CRITICAL INSTRUCTIONS:
        1. **LANGUAGE ADAPTATION:** If the user writes in Roman Urdu (e.g., 'net slow hai'), you MUST reply in Roman Urdu. 
        2. If the user writes in English, reply in English.
        3. **TONE:** Be extremely polite, empathetic, and professional. Use emojis 📡🛠️.
        4. **LIMIT:** Keep the response under 3-4 lines maximum.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return tech_solution