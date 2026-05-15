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

df = pd.read_csv('../data/fake_job_postings_cleaned.csv')

'''
    Selected columns for training
    [Text features]
    - title
    - company_profile
    - description
    - requirements
    - benefits
    [Target variable]
    - fraudulent 
'''

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
text_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
text_features = text_vectorizer.fit_transform(df['combined_text'])

# Combine all features
X = text_features
y = df['fraudulent']

result = []
best_model = None
best_model_name = None
best_f1 = 0
best_split = None

for split in splits:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        f1 = f1_score(y_test, pred, zero_division=0)
        
        result.append({
            'model': model_name,
            'split': split,
            'f1_score': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name
            best_split = split
        
result_df = pd.DataFrame(result)
print("Model Results: ")
print(result_df.sort_values(by='f1_score', ascending=False))

print(f"Best Model: {best_model_name} with F1 Score: {best_f1} at split: {best_split}")
# Modeling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_train)

# pred = model.predict(X_test)
    
joblib.dump(best_model, '../model/text_best_model.pkl')
joblib.dump(text_vectorizer, '../model/text_vectorizer.pkl')