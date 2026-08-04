from openai import OpenAI, OpenAIError, RateLimitError
import os

class LLMClient:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY env variable is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.temperature = 0.3
        self.max_tokens = 350

    def generate_response(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model,messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except RateLimitError:
            if "insufficient_quota" in str(e):
                return "Error: Insufficient quota for OpenAI API. Please check your usage and billing details."
            return "Error: API rate limit exceeded. Please try again later."
        except OpenAIError as e:
            print(f"OpenAI API Error: {str(e)}")
            return ("An error occured while communicating with the AI service")
        except Exception as e:
            print(f"Error: {str(e)}")
            return ("An unexpected error occured while generating the response.")