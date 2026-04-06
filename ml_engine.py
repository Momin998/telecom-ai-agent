# ml_engine.py
import pandas as pd
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from db_manager import load_data_fresh

def add_pakistani_noise(text):
    """Industry Secret: Intentionally corrupts text to mimic real Pakistani users."""
    if not isinstance(text, str):
        return str(text)
        
    variations = [
        text,  # Original
        text.replace("hai", "ha").replace("nahi", "nai"),
        text.replace("internet", "net").replace("aur", "ar"),
        text.replace("kya", "kyaaa").replace("hai", "h"),
        text.replace("bakwas", "farig").replace("chal", "chl"),
        text + " yr plz",
        text.replace("balance", "blnc").replace("kat", "cut")
    ]
    return random.choice(variations)

def train_bilingual_model():
    """Hybrid Data Training with Character N-Grams and Noise Injection."""
    print("🚀 Training Professional ML Model...")
    
    # 1. Load Real DB Data (From Supabase)
    df_real = load_data_fresh()
    
    # 2. Load Synthetic Baseline Data (The CSV we just created)
    if os.path.exists("training_data.csv"):
        df_base = pd.read_csv("training_data.csv")
    else:
        print("⚠️ Warning: training_data.csv not found! Model will be weak.")
        df_base = pd.DataFrame(columns=["text", "category"])
    
    # 3. Data Augmentation (Dynamic Noise Injection - x20 multiplier)
    noisy_data = []
    for _ in range(20): 
        for index, row in df_base.iterrows():
            noisy_data.append({
                "text": add_pakistani_noise(row["text"].lower()),
                "category": row["category"]
            })
            
    df_synthetic = pd.DataFrame(noisy_data)
    
    # 4. Combine Real + Augmented Data
    df_train = pd.concat([df_real, df_synthetic], ignore_index=True)
    df_train = df_train.dropna(subset=['text', 'category'])
    
    # 5. THE MASTERSTROKE: Character Level TF-IDF
    # char_wb with ngram (3,5) handles spelling mistakes flawlessly (e.g. ahista vs aista)
    model_pipeline = make_pipeline(
        TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)), 
        LogisticRegression(class_weight='balanced', max_iter=1000)
    )
    
    model_pipeline.fit(df_train['text'], df_train['category'])
    print(f"✅ Training Complete on {len(df_train)} augmented records.")
    
    return model_pipeline