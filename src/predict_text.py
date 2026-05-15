from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocess import clean_text
import joblib
from scipy.sparse import hstack
import pandas as pd

model = joblib.load('model/text_best_model.pkl')
vectorizer = joblib.load('model/text_vectorizer.pkl')

def predict_posting_text(text):
    cleaned_text = clean_text(text)
    text_features = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(text_features)[0]
    probability = model.predict_proba(text_features)[0]
    
    confidence = max(probability)
    
    if prediction == 1:
        label = 'Fraudulent'
    else:
        label = 'Not Fraudulent'
    
    return {
        'label': label,
        'confidence': confidence
    }
    