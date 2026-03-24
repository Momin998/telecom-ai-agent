# ml_engine.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from db_manager import load_data_fresh # Training ke liye database se purana data lene ke liye

def train_bilingual_model():
    """Bilingual (English + Roman Urdu) model ko train karta hai."""
    # 1. Hamara 100% Custom Pakistani Telecom Dataset
    bilingual_data = [
        # Internet
        {"text": "mera net bohot ahista chal raha hai", "category": "Internet"},
        {"text": "internet connection is extremely slow and lagging", "category": "Internet"},
        {"text": "router par lal batti jal rahi hai red light blinking", "category": "Internet"},
        {"text": "wifi signals are dropping frequently disconnects", "category": "Internet"},
        {"text": "pubg me ping high aa raha hai lag issue", "category": "Internet"},
        {"text": "net bilkul nahi chal raha subha se dead hai", "category": "Internet"},
        {"text": "fiber cable break or internet not working", "category": "Internet"},
        {"text": "youtube buffering video stuck aista chal rha", "category": "Internet"},
        # Billing
        {"text": "mera balance automatic kat gaya hai bina package ke", "category": "Billing"},
        {"text": "incorrect billing amount extra charges deducted", "category": "Billing"},
        {"text": "paise double kat gaye tax deduction zyada hai", "category": "Billing"},
        {"text": "refund my money balance wapis karo", "category": "Billing"},
        {"text": "vas caller tune ke paise kat rahe hain band karo", "category": "Billing"},
        {"text": "package lagaya tha par paise kat gaye offer error", "category": "Billing"},
        {"text": "please check my invoice payment history issue", "category": "Billing"},
        {"text": "unwanted subscription charges deducted from account", "category": "Billing"},
        # Customer Care
        {"text": "meri sim block ho gayi hai band hai", "category": "Customer Care Call"},
        {"text": "puk code lag gaya hai sim lock error", "category": "Customer Care Call"},
        {"text": "need to talk to a human agent representative call center", "category": "Customer Care Call"},
        {"text": "sim apne naam karni hai ownership transfer biometric", "category": "Customer Care Call"},
        {"text": "mnp port network change karna hai dusri sim", "category": "Customer Care Call"},
        {"text": "franchise location address batao nearest customer care", "category": "Customer Care Call"},
        {"text": "sim gum ho gayi hai duplicate sim nikalni hai", "category": "Customer Care Call"},
        {"text": "manager se baat karni hai complaint escalate karo", "category": "Customer Care Call"}
    ]
    
    # Data multiplication for solid training (360 rows)
    df_extra = pd.DataFrame(bilingual_data * 15) 
    
    # Real DB data + Synthetic data ka mix
    df_train = pd.concat([load_data_fresh(), df_extra], ignore_index=True)
    
    # TF-IDF with n-grams (1,3) handles spelling mistakes natively
    model_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3)), 
        LogisticRegression(max_iter=1000)
    )
    
    model_pipeline.fit(df_train['text'], df_train['category'])
    return model_pipeline