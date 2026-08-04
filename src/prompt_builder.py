

class PromptBuilder:

    WORD_LIMIT = 100

    @staticmethod
    def build_chat_prompt(context):
    
        feature_text = "\n".join(
            [
                f"{feature}: {value}"
                for feature, value in context["features"].items()
            ]
        )

        if context["shap_values"]:
            shap_text = "\n".join(
                [
                    f"- {row['feature_name']}: "
                    f"value={row['feature_value']}, "
                    f"SHAP={float(row['shap_value']):.4f}"
                    for row in context["shap_values"]
                ]
            )
        else:
            shap_text = "No SHAP values available."

        job_text = " ".join(
            context["input_text"].split()[:PromptBuilder.WORD_LIMIT]
        )

        return f"""
                You are an AI assistant that explains machine learning predictions for fraudulent job postings.

                Prediction:
                {context["prediction_label"]}

                Confidence:
                {context["confidence"]:.2%}

                Original Job Posting:
                {job_text}

                Structured Features:
                {feature_text}

                Top SHAP Features:
                {shap_text}

                Initial AI Explanation:
                {context["initial_explanation"]}

                Rules:

                1. Answer ONLY questions about this job posting.
                2. Use the SHAP values as the primary evidence when explaining predictions.
                3. Use the structured features when relevant.
                4. Use the original job posting text when relevant.
                5. Maintain conversation context across multiple questions.
                6. Never invent information that is not contained within the prediction context.
                7. If the user asks something unrelated to this prediction, politely explain that you can only discuss this prediction.
                8. Keep responses concise (approximately 150 words or fewer).
                """