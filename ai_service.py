# ai_service.py
import google.generativeai as genai
import streamlit as st

# 1. 🌟 Asal Enterprise Knowledge Base Import Karein
from knowledge_base import ISP_KNOWLEDGE_BASE

def detect_language(text):
    """
    Advanced Language Detection Engine.
    Differentiates between Pakistani Roman Urdu and Professional/Basic English.
    """
    text_lower = text.lower()
    # Aam Pakistani alfaz jo data mein use hue hain
    urdu_markers = [
        'hai', 'nahi', 'nai', 'yar', 'karo', 'masla', 'chal', 'raha', 
        'bhai', 'batti', 'kabhi', 'bohat', 'lag', 'kat', 'gaya', 'mera', 
        'mein', 'bhi', 'kya', 'kyun'
    ]
    
    urdu_count = sum(1 for word in urdu_markers if word in text_lower)
    
    # Agar 1 bhi Urdu ka lafz mil gaya toh yeh Roman Urdu hai, warna English
    if urdu_count >= 1:
        return "roman_urdu"
    else:
        return "english"

def get_rag_solution(user_text, category, mood="Neutral"):
    """
    🔥 FULL DEEP RAG (Retrieval-Augmented Generation) PIPELINE 🔥
    Gemini is highly restricted ONLY to the imported ISP Knowledge Base.
    """
    if "GEMINI_API_KEY" not in st.secrets:
        return "System API Key missing hai. Kripya backend admin se rabta karein."

    try:
        # Step 1: Language Detect Karo (Roman Urdu vs English)
        lang = detect_language(user_text)
        
        # Step 2: Database se Category uthao
        solutions = ISP_KNOWLEDGE_BASE.get(category, [])
        if not solutions:
            return "Maazrat, is waqt system mein is category ka solution update nahi hai. Kripya Helpline par call karein."
        
        # Step 3: Smart Trigger Matching (Retrieval)
        matched_solutions = []
        text_lower = user_text.lower()
        
        for sol in solutions:
            # Agar koi bhi trigger user ke text mein mil jaye
            if any(trigger in text_lower for trigger in sol["trigger"]):
                matched_solutions.append(sol)
        
        # 🧠 DEEP RAG FALLBACK: Agar user ne koi naya lafz use kiya jo trigger mein nahi hai, 
        # toh us category ke saare solutions AI ko bhej do taake AI khud soch kar hal nikal le.
        if not matched_solutions:
            matched_solutions = solutions
            
        # Step 4: Context Injection (Augmentation)
        context_text = ""
        for i, sol in enumerate(matched_solutions, 1):
            context_text += f"\n[Solution ID: {sol['id']}]\nUrdu Policy: {sol['solution_urdu']}\nEnglish Policy: {sol['solution_english']}\n"
            
        # Step 5: Strict Language Tone Setup
        if lang == "roman_urdu":
            tone_instruction = "IMPORTANT: You MUST reply ONLY in Pakistani Roman Urdu. Be polite, and use technical ISP terms clearly. DO NOT use Hindi words like kripya, kshama, or prabandh."
        else:
            tone_instruction = "IMPORTANT: You MUST reply ONLY in Professional Corporate English (or Basic English depending on user proficiency). Be polite and empathetic."

        # Step 6: Gemini Generation (Zero Hallucination Mode)
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are 'SmartFiber AI', a highly professional technical support agent for a top-tier Broadband Fiber ISP in Pakistan.

--- TICKET CONTEXT ---
Customer Complaint: "{user_text}"
System Detected Category: "{category}"
Customer Mood: "{mood}"
----------------------

--- APPROVED OFFICIAL SOLUTIONS (KNOWLEDGE BASE) ---
{context_text}
----------------------------------------------------

YOUR MISSION:
1. Carefully read the Customer Complaint.
2. Review the APPROVED OFFICIAL SOLUTIONS provided above.
3. Select the SINGLE most relevant solution matching the user's issue.
4. Explain the exact troubleshooting steps based ONLY on the selected solution.

CRITICAL ENTERPRISE GUARDRAILS (MUST OBEY):
- ZERO HALLUCINATION: DO NOT invent any troubleshooting steps, prices, IP addresses, or policies outside of the APPROVED OFFICIAL SOLUTIONS.
- TONE & LANGUAGE: {tone_instruction}
- PROMISES: DO NOT promise refunds, package upgrades, or free bandwidth.
- ESCALATION: If none of the solutions fit the complaint perfectly, politely apologize and advise the customer to call the helpline or book a technician visit.
- FORMATTING: Use bullet points for steps and appropriate emojis (📡, ⚙️, 💻, 💳).
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"RAG System Error: {e}")
        return "System network error. Kripya apna router restart karein ya helpline par call karein."


def fallback_intent_classifier(user_text):
    """
    Agar ML model fail ho jaye (confidence < 60%), toh Gemini Broadband ISP ke hisab se category batayega.
    """
    if "GEMINI_API_KEY" not in st.secrets:
        return "Internet" # Default fallback
        
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert AI Router for a Fiber Broadband ISP in Pakistan.
        Classify the following user complaint into exactly ONE of these three categories:
        1. Internet (For WiFi, router, fiber cut, speed, ping, gaming issues)
        2. Billing (For invoice, taxes, static IP charges, FUP limits, suspended account)
        3. Customer Care Call (For relocation, hardware replacement, admin panel access, technical support visits)
        
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
            return "Internet"
    except Exception as e:
        print(f"Fallback Classifier Error: {e}")
        return "Internet"