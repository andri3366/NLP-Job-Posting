"""Flask application entry point for prediction, persistence, and web routes."""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pickle
import joblib
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.predict import predict_posting, get_shap_values
from src.predict_text import predict_posting_text, get_text_shap_values
from src.llm_explain import explain_prediction
from auth import auth
from chatbot import chatbot
from supabase_client import supabase, secret_key

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key')

app.register_blueprint(auth, url_prefix='/auth')  # Register the auth blueprint
app.register_blueprint(chatbot)

# Load models
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model/tune_best_model.pkl')
vectorizer_path = os.path.join(base_dir, 'model/tune_best_extractor_vectorizer.pkl')
cat_features_path = os.path.join(base_dir, 'model/tune_cat_features.pkl')
text_model_path = os.path.join(base_dir, 'model/tune_text_best_model.pkl')
text_vectorizer_path = os.path.join(base_dir, 'model/tune_text_best_extractor.pkl')

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

def save_prediction_to_db(user_id, prediction_type, prediction_label, confidence, input_text, features=None, shap_values=None, prompt=None, explanation=None):
    try:

        if 'access_token' in session:
            supabase.auth.set_session(
                session.get('access_token'),
                session.get('refresh_token')
            )

        history_data = {
            "user_id": user_id,
            "prediction_type": prediction_type,  
            "prediction_label": prediction_label,
            "confidence": float(confidence),
            "input_text": input_text,
            "created_at": datetime.now().isoformat()
        }

        history_result = supabase.table("prediction_history").insert(history_data).execute()
        
        if history_result.data:
            prediction_id = history_result.data[0]['id']
            
            # If full model, save features
            if prediction_type == 'full' and features:
                features_data = {
                    "prediction_id": prediction_id,
                    "telecommuting": features.get('telecommuting') == 'Yes',
                    "has_company_logo": features.get('has_company_logo') == 'Yes',
                    "has_questions": features.get('has_questions') == 'Yes',
                    "employment_type": features.get('employment_type', 'Missing'),
                    "required_experience": features.get('required_experience', 'Missing'),
                    "required_education": features.get('required_education', 'Missing'),
                    "country": features.get('country', 'Missing')
                }
                supabase.table("prediction_features").insert(features_data).execute()
            
            # Save SHAP values if they exist
            if shap_values and len(shap_values) > 0:
                shap_data = []
                for shap in shap_values[:10]:  # Save top 10 SHAP values
                    shap_data.append({
                        "prediction_id": prediction_id,
                        "feature_name": shap.get('feature', ''),
                        "feature_value": str(shap.get('value', '')),
                        "shap_value": float(shap.get('shap_value', 0))
                    })
                if shap_data:
                    supabase.table("shap_results").insert(shap_data).execute()
            
            # Save AI explanation if provided
            # if prompt and explanation:
            #     explanation_data = {
            #         "prediction_id": prediction_id,
            #         "prompt": prompt,
            #         "explanation": explanation,
            #         "created_at": datetime.now().isoformat()
            #     }
            #     supabase.table("ai_explanations").insert(explanation_data).execute()
            
            return prediction_id
    
    except Exception as e:
        print(f"Error saving to database: {e}")
        import traceback
        traceback.print_exc()
        return None
   
_session_cleared = False

@app.before_request
def clear_session_on_startup():
    global _session_cleared
    if not _session_cleared:
        session.clear()
        _session_cleared = True
        print("Session cleared on app startup")

@app.route('/')
def index():
    if 'access_token' in session:
        return render_template('index.html', logged_in=True)
    return render_template('index.html', logged_in=False)

@app.route('/full_model', methods=['GET', 'POST'])
def full_model():

    if 'access_token' not in session:
        flash('Please login to access this feature.', 'warning')
        return redirect(url_for('auth.login'))
    
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
                print(type(shap_df))
                # Convert to list for JSON serialization if it's a DataFrame
                if isinstance(shap_df, pd.DataFrame):
                    shap_df = shap_df.rename(columns={"Feature" : "feature", "Value" : "value", "SHAP" : "shap_value"}).to_dict('records')
                    # shap_df = shap_df.to_dict('records')

            user_id = session.get('user_id')
            if user_id:
                prediction_id = save_prediction_to_db(
                    user_id=user_id,
                    prediction_type='full',
                    prediction_label=result.get('label'),
                    confidence=result.get('confidence'),
                    input_text=combined_text,
                    features=features,
                    shap_values=shap_df
                )
                session["prediction_id"] = prediction_id
                flash('Prediction saved to your history!', 'success')

            # Store only needed sessions
            session['last_label'] = result.get('label')
            session['last_text'] = combined_text
            session['last_features'] = features
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
                         explanation=explanation,
                         logged_in=True)

@app.route('/text_model', methods=['GET', 'POST'])
def text_model():
    result = None
    explanation = None
    shap_df = None
    text = ""

    is_logged_in = 'access_token' in session

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
                        shap_df = shap_df.rename(columns={"Feature" : "feature", "Value" : "value", "SHAP" : "shap_value"}).to_dict('records')
                        # shap_df = shap_df.to_dict('records')
                        # shap_df = shap_df.rename(columns={"Feature" : "feature", "SHAP" : "shap_value"}).to_dict('records')

                if is_logged_in:
                    user_id = session.get('user_id')
                    if user_id:
                        prediction_id = save_prediction_to_db(
                            user_id=user_id,
                            prediction_type='text',
                            prediction_label=result.get('label'),
                            confidence=result.get('confidence'),
                            input_text=text,
                            shap_values=shap_df
                        )
                        session["prediction_id"] = prediction_id
                        flash('Prediction saved to your history!', 'success')
                else:
                    flash('Prediction complete! Login to save your history.', 'info')

                # Store what's needed
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
                         explanation=explanation,
                         logged_in = is_logged_in)

@app.route('/get_explanation', methods=['POST'])
def get_explanation():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        mode = data.get('mode', 'full')
        prediction_id = session.get("prediction_id")
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

        # if 'access_token' in session:
        #     user_id = session.get('user_id')
        #     # Get the last prediction ID from the database
        #     try:
        #         last_prediction = supabase.table("prediction_history")\
        #             .select("id")\
        #             .eq("user_id", user_id)\
        #             .order("created_at", desc=True)\
        #             .limit(1)\
        #             .execute()
                
        #         if last_prediction.data:
        #             prediction_id = last_prediction.data[0]['id']
        #             # Save explanation
        #             explanation_data = {
        #                 "prediction_id": prediction_id,
        #                 "prompt": prompt,
        #                 "explanation": explanation,
        #                 "created_at": datetime.now().isoformat()
        #             }
        #             supabase.table("ai_explanations").insert(explanation_data).execute()
        #     except Exception as e:
        #         print(f"Error saving explanation: {e}")

        if prediction_id:
            try:
                explanation_data = {
                    "prediction_id" : prediction_id,
                    "prompt" : prompt,
                    "explanation" : explanation,
                    "created_at" : datetime.now().isoformat()
                }

                supabase.table("ai_explanations").insert(explanation_data).execute()
            except Exception as e:
                print(f"Error saving explanation: {e}")

        return jsonify({"success": True, 'explanation': explanation, "prediction_id": prediction_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    if 'access_token' not in session:
        flash('Please login to view your history.', 'warning')
        return redirect(url_for('auth.login'))
    
    try:

        if 'access_token' in session:
            supabase.auth.set_session(
                session.get('access_token'),
                session.get('refresh_token')
            )

        user_id = session.get('user_id')
        
        if not user_id:
            flash('User ID not found. Please login again.', 'warning')
            return redirect(url_for('auth.login'))
        
        # Get prediction history
        predictions = supabase.table("prediction_history")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        
        predictions_data = predictions.data if predictions.data else []

        print("\nPrediction Labels:")
        for pred in predictions_data:
            print(pred["prediction_label"])
        # Calculate analytics
        analytics = {
            'total_predictions': len(predictions_data),
            'full_model_count': 0,
            'text_model_count': 0,
            'fake_count': 0,
            'real_count': 0,
            'total_confidence': 0,
            'avg_confidence': 0,
            'fake_avg_confidence': 0,
            'real_avg_confidence': 0,
            'latest_prediction': predictions_data[0] if predictions_data else None,
            'most_used_model': 'N/A',
            'fake_percentage': 0,
            'real_percentage': 0
        }

        fake_conf_total = 0
        real_conf_total = 0

        if predictions_data:
            for pred in predictions_data:
                if pred.get('prediction_type') == 'full':
                    analytics['full_model_count'] += 1
                else:
                    analytics['text_model_count'] += 1
                
                confidence = float(pred.get("confidence", 0))

                analytics["total_confidence"] += confidence

                label = pred.get("prediction_label", "").strip().lower()

                if label in ["fake", "fraudulent"]:
                    analytics["fake_count"] += 1
                    fake_conf_total += confidence

                elif label in ["real", "not fraudulent"]:
                    analytics["real_count"] += 1
                    real_conf_total += confidence
            
            total = analytics['total_predictions']
            if total > 0:

                analytics["avg_confidence"] = (
                    analytics["total_confidence"] / total
                )

                analytics["fake_percentage"] = (
                    analytics["fake_count"] / total
                ) * 100

                analytics["real_percentage"] = (
                    analytics["real_count"] / total
                ) * 100

                if analytics["fake_count"] > 0:
                    analytics["fake_avg_confidence"] = (
                        fake_conf_total / analytics["fake_count"]
                    )

                if analytics["real_count"] > 0:
                    analytics["real_avg_confidence"] = (
                        real_conf_total / analytics["real_count"]
                    )

            if analytics["full_model_count"] > analytics["text_model_count"]:
                analytics["most_used_model"] = "Full Model"
            elif analytics["text_model_count"] > analytics["full_model_count"]:
                analytics["most_used_model"] = "Text Model"
            else:
                analytics["most_used_model"] = "Both Equally"
            
            if predictions_data:
                analytics['latest_prediction'] = predictions_data[0]
        
        return render_template('history.html', 
                             predictions=predictions_data,
                             analytics=analytics,
                             logged_in=True)
                             
    except Exception as e:
        flash(f'Error loading history: {str(e)}', 'danger')
        return render_template('history.html', 
                             predictions=[],
                             analytics=None,
                             logged_in=True)
    
if __name__ == "__main__":
    app.run(debug=False)