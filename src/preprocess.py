# Raw text collection
# Noise removal
# Normalization
# Tokenization
# Stop word removal
# Stemming and Lemmatization
# Final Output for vectorization
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Missing values
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove noise
    tokens = [re.sub(r'[^A-Za-z0-9]+', '', token) for token in tokens]
    
    # Remove empty tokens
    tokens = [token for token in tokens if token]
    
    # Stop word removal
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

  