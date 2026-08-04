from openai import OpenAI, RateLimitError, OpenAIError
import os
from src.llm_client import LLMClient
# from src.hidden import key

# key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=key)
llm = LLMClient()

def explain_prediction(prediction, text, prompt, features, shap_values):
    

    word_limit = 100
    feature_text = "\n".join(
        [f"{feature}: {value}" for feature, value in features.items()]
    )
    if shap_values:
        shap_text = "\n".join(
            [
                f"- {row['feature']}: value={row['value']}, SHAP={row['shap_value']:.4f}"
                for row in shap_values
            ]
        )
    else:
        shap_text = "No SHAP values available."

    job_text = " ".join(
        text.split()[:word_limit]
    )

    messages=[{"role": "user", "content": 
                f"""
                Prediction:
                {prediction}

                Job Posting:
                {job_text}

                Structured Features:
                {feature_text}

                Top SHAP Features:
                {shap_text}

                Additional Prompt:
                {prompt}
                """},
            {"role": "system", "content": 
                """
                You are an AI assistant that explains why a machine learning model classified a job posting as fraudulent or legitimate.

                Explain the prediction using:

                • The SHAP feature importance
                • The job posting text
                • The structured features

                Do not invent information.
                Keep the explanation concise.
                """}
            ]
    
    return llm.generate_response(messages)
