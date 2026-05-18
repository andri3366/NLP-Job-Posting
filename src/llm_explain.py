from openai import OpenAI, RateLimitError, OpenAIError
import os
# from src.hidden import key

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def explain_prediction(prediction, text, prompt):
    
    if not key:
        return "Error: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
    try:
        
        word_limit = 100
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Explain why the model predicted '{prediction}':\n\n{' '.join(text.split()[:word_limit])}\n\n{prompt}"},
                  {"role": "system", "content": "You are an AI assistant that provides explanations and analysis only for why the machine learning model classified the job posting as fraudulent or not fraudulent based on the specified feature."
                   "Focus on the key words and phrases in the text that influenced the model's decision. Do not provide general information about unrelated details."}
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