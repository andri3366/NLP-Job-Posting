from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pickle
import joblib
import os
import json
import pandas as pd
import numpy as np

from src.predict import predict_posting, get_shap_values
from src.predict_text import predict_posting_text, get_text_shap_values
from src.llm_explain import explain_prediction

app = Flask(__name__)
app.secret_key = 'place_holder'

# Load models
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model/best_model.pkl')
vectorizer_path = os.path.join(base_dir, 'model/best_extractor_vectorizer.pkl')
cat_features_path = os.path.join(base_dir, 'model/cat_features.pkl')
text_model_path = os.path.join(base_dir, 'model/text_best_model.pkl')
text_vectorizer_path = os.path.join(base_dir, 'model/text_best_extractor.pkl')

# Caching models to load once
model = None
vectorizer = None
cat_columns = None
text_model = None
text_vectorizer = None

# Load resources
def load_full_resources():
    global model, vectorizer, cat_columns
    if model is None:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        cat_columns = joblib.load(cat_features_path)

    return model, vectorizer, cat_columns

def load_text_resources():
    global text_model, text_vectorizer
    if text_model is None:
        text_model = joblib.load(text_model_path)
        text_vectorizer = joblib.load(text_vectorizer_path)

    return text_model, text_vectorizer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/full_model', methods=['GET', 'POST'])
def full_model():
    result = None
    explanation = None
    shap_df = None
    features = {}
    text = ""
    
    if request.method == 'POST':
        try:
            title = request.form.get('title', '')
            company_profile = request.form.get('company_profile', '')
            text = request.form.get('text', '')
            requirements = request.form.get('requirements', '')
            benefits = request.form.get('benefits', '')
            
            telecommuting = request.form.get('telecommuting', 'No')
            has_company_logo = request.form.get('has_company_logo', 'No')
            has_questions = request.form.get('has_questions', 'No')
            employment_type = request.form.get('employment_type', 'Missing')
            required_education = request.form.get('required_education', 'Missing')
            required_experience = request.form.get('required_experience', 'Missing')      
            country = request.form.get('country', 'Missing')
            
            # Load model
            model, vectorizer, cat_columns = load_full_resources()
            
            combined_text = (title + ' ' + company_profile + ' ' + text + ' ' + 
                           requirements + ' ' + benefits)
            
            result = predict_posting(
                combined_text,
                telecommuting,
                has_company_logo,
                has_questions,
                employment_type,
                required_experience,
                required_education,
                country
            )  
            
            # Store features for explanation
            features = {
                "telecommuting": telecommuting,
                "has_company_logo": has_company_logo,
                "has_questions": has_questions,
                "employment_type": employment_type,
                "required_experience": required_experience,
                "required_education": required_education,
                "country": country
            }
            
            # Get SHAP values for display
            X = result.get('X')
            if X is not None:
                shap_df = get_shap_values(X)
                # Convert to list for JSON serialization if it's a DataFrame
                if isinstance(shap_df, pd.DataFrame):
                    shap_df = shap_df.to_dict('records')
            
            # Store ONLY what's needed for explanation in session
            # DON'T store the entire result with csr_matrix
            session['last_label'] = result.get('label')
            session['last_text'] = combined_text
            session['last_features'] = features
            # Store shap as list of dicts (already converted above)
            session['last_shap'] = shap_df
            
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            print("Full error details:", e)
            import traceback
            traceback.print_exc()
    
    return render_template('full_model.html', 
                         result=result, 
                         text=text,
                         features=features,
                         shap_df=shap_df,
                         explanation=explanation)

@app.route('/text_model', methods=['GET', 'POST'])
def text_model():
    result = None
    explanation = None
    shap_df = None
    text = ""
    
    if request.method == 'POST':
        try:
            text = request.form.get('text', '')
            
            if not text.strip():
                flash('Please enter a job description.', 'warning')
            else:
                # Load text model
                text_model, text_vectorizer = load_text_resources()
                
                # Get prediction
                result = predict_posting_text(text)
                
                # Get SHAP values
                X = result.get('X')
                if X is not None:
                    shap_df = get_text_shap_values(X)
                    if isinstance(shap_df, pd.DataFrame):
                        shap_df = shap_df.to_dict('records')
                
                # Store ONLY what's needed
                session['last_label'] = result.get('label')
                session['last_text'] = text
                session['last_shap'] = shap_df
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            print("Full error details:", e)
            import traceback
            traceback.print_exc()
    
    return render_template('text_model.html', 
                         result=result, 
                         text=text,
                         shap_df=shap_df,
                         explanation=explanation)

@app.route('/get_explanation', methods=['POST'])
def get_explanation():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        mode = data.get('mode', 'full')
        
        # Get stored data from session
        label = session.get('last_label')
        text = session.get('last_text', '')
        features = session.get('last_features', {})
        shap_df = session.get('last_shap', [])
        
        if not label:
            return jsonify({'error': 'No prediction found'}), 400
        
        # Get explanation
        explanation = explain_prediction(
            prediction=label,
            text=text,
            prompt=prompt,
            features=features,
            shap_values=shap_df
        )
        
        return jsonify({'explanation': explanation})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)