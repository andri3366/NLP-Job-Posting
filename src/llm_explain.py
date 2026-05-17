from openai import OpenAI, RateLimitError
import os
# from src.hidden import key

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def explain_prediction(prediction, text, prompt):
    
    try:
        
        word_limit = 100
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Explain why the model predicted '{prediction}':\n\n{' '.join(text.split()[:word_limit])}\n\n{prompt}"},
                  {"role": "system", "content": "You are an AI assistant that provides explanations and analysis only for why the machine learning model classified the job posting as fraudulent or not fraudulent based on the specified feature."
                   "Focus on the key words and phrases in the text that influenced the model's decision. Do not provide general information about unrelated details."}
                  ]
        )
        return response.choices[0].message.content.strip()
    except RateLimitError:
        return "Error: API rate limit exceeded. Please try again later."
    except Exception as e:
        return f"Error: {str(e)}"