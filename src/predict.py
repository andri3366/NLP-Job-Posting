from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocess import clean_text
import joblib
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import shap
from pathlib import Path
# model = joblib.load('model/best_model.pkl')
# # vectorizer = joblib.load('model/vectorizer.pkl')
# vectorizer = joblib.load('model/best_extractor_vectorizer.pkl')
# cat_columns = joblib.load('model/cat_features.pkl')

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

model = joblib.load(MODEL_DIR / "best_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "best_extractor_vectorizer.pkl")
cat_columns = joblib.load(MODEL_DIR / "cat_features.pkl")
def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    conf_matrix = confusion_matrix(y_test, pred)
    
    return pred, accuracy, report, conf_matrix

# Used the tree explainer as it is optimized for XGBoost
explainer = shap.TreeExplainer(model)
print("Model expects:", model.n_features_in_)

def get_shap_values(X):
    print("X shape:", X.shape)
    # explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)

    feature_names = (
        # list only works because of the us of TF-IDF
        list(vectorizer.get_feature_names_out()) + [
            "telecommuting",
            "has_company_logo",
            "has_questions"
        ] +
        list(cat_columns)
    )

    shap_df = pd.DataFrame({"Feature" : feature_names, "SHAP" : shap_values.values[0]})

    shap_df["Importance"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values(by="Importance", ascending=False)

    return shap_df.head(10)

def predict_posting(text, telecommuting, has_company_logo, has_questions, employment_type, required_experience, required_education, country):
    cleaned_text = clean_text(text)
    text_features = vectorizer.transform([cleaned_text])
    
    binary_features = pd.DataFrame({
        'telecommuting': [telecommuting],
        'has_company_logo': [has_company_logo],
        'has_questions': [has_questions]
    })
    
    # scipy.sparse requires numeric, therefore I converted the binary and categorical feature to int
    binary_features = binary_features.replace({'Yes': 1, 'No': 0})
    binary_features = binary_features.astype(int)
    
    cat_features = pd.DataFrame({
        'employment_type': [employment_type],
        'required_experience': [required_experience],
        'required_education': [required_education],
        'country': [country]
    })
    
    cat_features_encoded = pd.get_dummies(cat_features, drop_first=True)
    cat_features_training = cat_features_encoded.reindex(columns=cat_columns, fill_value=0)
    cat_features_training = cat_features_training.astype(int)
    
    # needed to sparse the values as XGBoost often requires consistent sparse types
    binary_features_sparse = csr_matrix(binary_features.values)
    cat_features_sparse = csr_matrix(cat_features_training.values)
    
    X = hstack([text_features, binary_features_sparse, cat_features_sparse])
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # top_shap = get_shap_values(X) if explain else None
    
    confidence = max(probability)
    
    if prediction == 1:
        label = 'Fraudulent'
    else:
        label = 'Not Fraudulent'
    
    return {
        'label': label,
        'confidence': confidence,
        # "shap" : top_shap
        'X' : X
    }
    