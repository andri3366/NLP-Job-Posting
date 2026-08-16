"""TF-IDF, SBERT, and hybrid feature extraction for model training and inference."""

import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

BINARY_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

CAT_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "country"
]

class TFIDFTextExtractor:

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

    def fit_transform(self,text):
        return self.vectorizer.fit_transform(text)

    def transform(self,text):
        return self.vectorizer.transform(text)
    
class SBERTTextExtractor:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, text):

        return self.model.encode(
            text.tolist(),
            show_progress_bar=True
        )

    def transform(self, text):

        return self.model.encode(
            text.tolist(),
            show_progress_bar=False
        )


class HybridTFIDFExtractor:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.cat_columns = None

    def fit_transform(self, df):

        text = self.vectorizer.fit_transform(df["combined_text"])

        binary = df[BINARY_COLUMNS]

        cat = pd.get_dummies(df[CAT_COLUMNS], drop_first=True)

        self.cat_columns = cat.columns

        return hstack([text, csr_matrix(binary.values), csr_matrix(cat.values)])

    def transform(self, df):

        text = self.vectorizer.transform(df["combined_text"])

        binary = df[BINARY_COLUMNS]

        cat = pd.get_dummies(df[CAT_COLUMNS], drop_first=True)

        cat = cat.reindex(columns=self.cat_columns, fill_value=0)

        return hstack([text, csr_matrix(binary.values), csr_matrix(cat.values)])


class HybridSBERTExtractor:

    def __init__(self,model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cat_columns = None

    def fit_transform(self, df):

        text = self.model.encode(df["combined_text"].tolist(), show_progress_bar=True)

        binary = df[BINARY_COLUMNS].values

        cat = pd.get_dummies(df[CAT_COLUMNS], drop_first=True)

        self.cat_columns = cat.columns

        return hstack([csr_matrix(text), csr_matrix(binary), csr_matrix(cat.values)])

    def transform(self, df):

        text = self.model.encode(df["combined_text"].tolist(), show_progress_bar=False)

        binary = df[BINARY_COLUMNS].values

        cat = pd.get_dummies(df[CAT_COLUMNS],drop_first=True)

        cat = cat.reindex(columns=self.cat_columns, fill_value=0)

        return hstack([csr_matrix(text),csr_matrix(binary),csr_matrix(cat.values)])