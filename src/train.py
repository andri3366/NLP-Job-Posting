import pandas as pd
import joblib

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score 
from src.preprocess import clean_text

# Test extractors
from feature_extractors import HybridTFIDFExtractor, HybridSBERTExtractor
df = pd.read_csv('../data/fake_job_postings_cleaned.csv')

'''
    Selected columns for training
    [Text features]
    - title
    - company_profile
    - description
    - requirements
    - benefits
    [Binary features]
    - telecommuting
    - has_company_logo
    - has_questions
    [Categorical features]
    - enrollment_type
    - required_experience
    - required_education
    - country
    [Target variable]
    - fraudulent 
'''
extractors = {
    "TF-IDF + Structured": HybridTFIDFExtractor(),
    "SBERT + Structured" : HybridSBERTExtractor()
}

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=1, n_estimators=100, random_state=42),
    "Naive Bayes": MultinomialNB()
                             
}

splits = [0.3, 0.2, 0.15]

# Combine the text
df['combined_text'] = (
    df['title'] + ' ' +
    df['company_profile'] + ' ' +
    df['description'] + ' ' +
    df['requirements'] + ' ' +
    df['benefits']
)

df['combined_text'] = df['combined_text'].apply(clean_text)

# Apply TF-IDF
# vectorizer = TfidfVectorizer(max_features=1000)
# text_features = vectorizer.fit_transform(df['combined_text'])

# Handle binary features
binary_features = df[['telecommuting', 'has_company_logo', 'has_questions']]

# Handle categorical features
cat_columns = ['employment_type', 'required_experience', 'required_education', 'country']
cat_features = pd.get_dummies(df[cat_columns], drop_first=True)


# Combine all features
# X = hstack([text_features, binary_features, cat_features])
y = df['fraudulent']

result = []
best_model = None
best_model_name = None
best_f1 = 0
best_split = None
best_extractor = None
best_extractor_name = None

for extractor_name, extractor in extractors.items():
    print(f"\n Testing {extractor_name}")

    X = extractor.fit_transform(df) 
    for split in splits:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
        
        for model_name, model in models.items():

            # Prevent incompatibility with neg values
            if (extractor_name == "SBERT + Structured" and model_name == "Naive Bayes"):
                continue

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            f1 = f1_score(y_test, pred, zero_division=0)
            
            result.append({
                'extractor': extractor_name,
                'model': model_name,
                'split': split,
                'f1_score': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = model_name
                best_split = split
                best_extractor = extractor
                best_extractor_name = extractor_name
        
result_df = pd.DataFrame(result)
print("Model Results: ")
print(result_df.sort_values(by='f1_score', ascending=False))

print(f"Best Model: {best_model_name} with F1 Score: {best_f1} at split: {best_split} with extractor: {best_extractor_name}")
# Modeling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_train)

# pred = model.predict(X_test)
    
joblib.dump(best_model, '../model/best_model.pkl')

if best_extractor_name == "TF-IDF + Structured":
    joblib.dump(best_extractor.vectorizer, '../model/best_extractor_vectorizer.pkl')
    joblib.dump(best_extractor.cat_columns,"../model/cat_features.pkl")
    joblib.dump(
        {
            "extractor": "tfidf"
        },
        "../model/feature_info.pkl"
    )

else:
    joblib.dump(
        best_extractor.cat_columns,
        "../model/cat_features.pkl"
    )

    joblib.dump(
        {
            "extractor": "sbert",
            "model_name": "all-MiniLM-L6-v2"
        },
        "../model/feature_info.pkl"
    )
# joblib.dump(vectorizer, '../model/vectorizer.pkl')
# joblib.dump(cat_features.columns, '../model/cat_features.pkl')