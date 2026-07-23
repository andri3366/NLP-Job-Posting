from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocess import clean_text
import joblib
from scipy.sparse import hstack
import pandas as pd
from src.predict import get_shap_values
import shap
from pathlib import Path

# model = joblib.load('../model/text_best_model.pkl')
# # vectorizer = joblib.load('model/text_vectorizer.pkl')
# vectorizer = joblib.load('../model/text_best_extractor.pkl')

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

model = joblib.load(MODEL_DIR / "text_best_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "text_best_extractor.pkl")

print(type(model))
print(type(vectorizer))
print(type(vectorizer).__name__)

text_explainer = shap.TreeExplainer(model)
def get_text_shap_values(X):

    shap_values = text_explainer(X)
    
    feature_names = vectorizer.vectorizer.get_feature_names_out()
    
    shap_df = pd.DataFrame({"Feature" : feature_names, "SHAP" : shap_values.values[0]})
    
    shap_df["Importance"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values(by="Importance", ascending=False)
    
    return shap_df.head(10)



def predict_posting_text(text):
    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])

    print(X.shape)
    
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
    