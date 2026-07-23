from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocess import clean_text
import joblib
from scipy.sparse import hstack
import pandas as pd
from src.predict import get_shap_values

from pathlib import Path

# model = joblib.load('../model/text_best_model.pkl')
# # vectorizer = joblib.load('model/text_vectorizer.pkl')
# vectorizer = joblib.load('../model/text_best_extractor.pkl')

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

model = joblib.load(MODEL_DIR / "text_best_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "text_best_extractor.pkl")

def predict_posting_text(text):
    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    confidence = max(probability)
    
    if prediction == 1:
        label = 'Fraudulent'
    else:
        label = 'Not Fraudulent'
    
    return {
        'label': label,
        'confidence': confidence,
        # 'shap': get_shap_values(X) if explain else None
        'X' : X
    }
    