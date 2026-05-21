from openai import OpenAI, RateLimitError, OpenAIError
import os
# from src.hidden import key

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def explain_prediction(prediction, text, prompt, features):
    
    if not key:
        return "Error: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
    try:
        
        word_limit = 100
        feature_text = "\n".join(
            [f"{feature}: {value}" for feature, value in features.items()]
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Explain why the model predicted '{prediction}':\n\n" f"Job Posting Text:{' '.join(text.split()[:word_limit])}\n\n"f"Structured features: {feature_text}\n\n"f"{prompt}\n\n"},
                  {"role": "system", "content": "You are an AI assistant that provides explanations and analysis only for why the machine learning model classified the job posting as fraudulent or not fraudulent based on both the specified job posting text and the structured binary and categorical features"
                   "Focus on the key words and phrases in the text and the structured features relavant to the prediction. Do not provide general information about unrelated details, only focus on the prediction using the text content and the structured features."}
                  ]
        )
        return response.choices[0].message.content.strip()
    except RateLimitError:
        if "insufficient_quota" in str(e):
            return "Error: Insufficient quota for OpenAI API. Please check your usage and billing details."
        return "Error: API rate limit exceeded. Please try again later."
    except OpenAIError as e:
        return f"OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"