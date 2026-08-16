"""Conversation service for retrieving prediction context and generating replies."""

from datetime import datetime
from supabase_client import supabase
from src.llm_client import LLMClient
from src.prompt_builder import PromptBuilder

class ChatbotServices:

    MAX_INTERACTIONS = 5

    def __init__(self):
        self.supabase = supabase
        self.llm = LLMClient()

    def start_conversation(self, prediction_id, user_id):
        #1
        try:
            prediction = (
                self.supabase.table("prediction_history").select("id").eq("id", prediction_id).eq("user_id", user_id).execute()
            )

            if not prediction.data:
                return {
                    "success" : False,
                    "message" : "Prediction not found."
                }

            existing = (
                self.supabase.table("conversations").select("id").eq("prediction_id", prediction_id).eq("user_id", user_id).limit(1).execute()
            )

            if existing.data:
                return {
                    "success" : True,
                    "conversation_id" : existing.data[0]["id"],
                    "message" : "Conversation already exists."
                }

            conversation = (
                self.supabase.table("conversations").insert({
                    "prediction_id" : prediction_id,
                    "user_id" : user_id,
                    "interaction_count" : 0
                }).execute()
            )

            conversation_id = conversation.data[0]["id"]

            # explanation = (
            #     self.supabase.table("ai_explanations").select("explanation").eq("prediction_id", prediction_id).order("created_at", desc=True).limit(1).execute()
            # )

            # if explanation.data : 
            #     print(self.supabase.auth.get_user())
            #     self._save_message(
            #         conversation_id=conversation_id,
            #         role="assistant",
            #         message=explanation.data[0]["explanation"]
            #     )

            return {
                "success" : True,
                "conversation_id" : conversation_id,
                "message" : "Conversation created successfully"
            }

        except Exception as e:
            return {
                "success" : False,
                "message" : str(e)
            }

    def ask_question(self, conversation_id, user_id, question):
        #6
        try:
            conversation = (
                self.supabase.table("conversations").select("*").eq("id", conversation_id).eq("user_id", user_id).single().execute()
            )

            if not conversation.data:
                return {
                    "success": False,
                    "message" : "Conversation not found."
                }

            conversation = conversation.data

            if conversation["interaction_count"] >= self.MAX_INTERACTIONS:
                return {
                    "success" : False,
                    "message" : "Reached the numbner of follow-up questions for this prediction."
                }

            context = self._load_prediction_context(conversation["prediction_id"], user_id)

            if context is None:
                return {
                    "success" : False,
                    "message" : "Unable to load prediction context"
                }

            history = self._load_messages(conversation_id)
            messages = self._build_openai_messages(context=context, history=history, question=question)

            assistant_response = self._call_llm(messages)

            if assistant_response is None:
                raise Exception("LLM returned no response.")
            
            self._save_message(conversation_id=conversation_id, role="user", message=question)

            if not self._save_message(conversation_id=conversation_id, role="user", message=question):
                raise Exception("Unable to save user message.")

            self._save_message(conversation_id=conversation_id, role="assistant", message=assistant_response)

            self._increment_counter(conversation_id)
            remaining = self.get_remaining_questions(conversation_id)

            return {
                "success" : True,
                "response" : assistant_response,
                "remaining_questions" : remaining,
                "interaction_count" : conversation["interaction_count"] + 1
            }

        except Exception as e:
            print(f"Chatbot error: {e}")

            return {
                "success" : False,
                "message" : str(e)
            }

    def get_conversation(self, conversation_id, user_id):
        #8
        try:

            conversation = (
                self.supabase.table("conversations").select("*").eq("id", conversation_id).eq("user_id", user_id).single().execute()
            )

            if not conversation.data:

                return {
                    "success": False,
                    "message": "Conversation not found."
                }

            conversation = conversation.data

            messages = (
                self.supabase.table("conversation_messages").select("*").eq("conversation_id", conversation_id).order("id").execute()
            )

            remaining = (
                self.MAX_INTERACTIONS - conversation["interaction_count"]
            )

            return {
                "success": True,
                "conversation_id": conversation["id"],
                "prediction_id": conversation["prediction_id"],
                "interaction_count": conversation["interaction_count"],
                "remaining_questions": remaining,
                "messages": messages.data
            }

        except Exception as e:

            print(f"Conversation error: {e}")
            return {
                "success": False,
                "message": str(e)
            }

    def get_remaining_questions(self, conversation_id):
        #9
        try:
            conversation = (
                self.supabase.table("conversations").select("interaction_count").eq("id", conversation_id).single().execute()
            )

            if not conversation.data:
                return 0

            interaction_count = conversation.data["interaction_count"]
            remaining = self.MAX_INTERACTIONS - interaction_count

            return max(0, remaining)
        except Exception as e:
            print(f"Remaing questions error: {e}")
            return 0

    def _load_prediction_context(self, prediction_id, user_id):
        #2

        try:

            prediction = (
                self.supabase.table("prediction_history").select("*").eq("id", prediction_id).eq("user_id", user_id).single().execute()
            )

            if not prediction.data:
                return None

            prediction_data = prediction.data

            features = (
                self.supabase.table("prediction_features").select("*").eq("prediction_id", prediction_id).execute()
            )

            if features.data:
                feature_data = features.data[0]
                feature_data.pop("id", None)
                feature_data.pop("prediction_id", None)
            else:
                feature_data = {}

            shap = (
                self.supabase.table("shap_results").select("*").eq("prediction_id", prediction_id).order("shap_value", desc=True).execute()
            )

            shap_data = shap.data if shap.data else []

            explanation = (
                self.supabase.table("ai_explanations").select("explanation").eq("prediction_id", prediction_id).order("created_at", desc=True).limit(1).execute()
            )

            if explanation.data:
                explanation_text = explanation.data[0]["explanation"]
            else:
                explanation_text = ""

            return {
                "prediction_id": prediction_data["id"],
                "prediction_label": prediction_data["prediction_label"],
                "prediction_type": prediction_data["prediction_type"],
                "confidence": prediction_data["confidence"],
                "input_text": prediction_data["input_text"],
                "created_at": prediction_data["created_at"],
                "features": feature_data,
                "shap_values": shap_data,
                "initial_explanation": explanation_text
            }

        except Exception as e:

            print(f"Prediction context error: {e}")

            return None            
        

    def _load_messages(self, conversation_id):
        #4

        try:
            # for future implementations when dealing with large amount of querying maybe order by id as well
            result = (
                self.supabase.table("conversation_messages").select("role, message").eq("conversation_id", conversation_id).order("id").order("created_at").execute()
            )

            message = []

            for row in result.data:
                message.append({
                    "role" : row["role"],
                    "content" : row["message"]
                })

            return message
        except Exception as e:
            print(f"Load messages error: {e}")
            return []

    def _save_message(self, conversation_id, role, message):
        #3
        valid_role = {"system", "assistant", "user"}
        try:

            if role not in valid_role:
                raise ValueError(f"Invalid role: {role}")
            
            self.supabase.table("conversation_messages").insert({
                "conversation_id" : conversation_id,
                "role" : role,
                "message" : message
            }).execute()

            return True

        except Exception as e:
            print(f"Save message error: {e}")
            return False
        

    def _increment_counter(self, conversation_id):
        #7
        try:
            conversation = (
                self.supabase.table("conversations").select("interaction_count").eq("id", conversation_id).single().execute()
            )

            if not conversation.data:
                return False

            current_count = conversation.data["interaction_count"]

            (
                self.supabase.table("conversations").update({
                    "interaction_count" : current_count + 1
                }).eq("id", conversation_id).execute()
            )

            return True
        except Exception as e:
            print(f"Increment counter error: {e}")
            return False
        

    def _build_openai_messages(self, context, history, question):

        # system_prompt = PromptBuilder.build_chat_prompt(context)

        # messages = [ {
        #     "role" : "system",
        #     "content": system_prompt
        # }]

        messages = [ {
            "role" : "system",
            "content" : PromptBuilder.build_chat_prompt(context)
            }]

        messages.extend(history)
        messages.append({
            "role" : "user",
            "content" : question
        })
        #5
        return messages

    def _call_llm(self, messages):
        #10
        try:
            return self.llm.generate_response(messages)
        except Exception as e:
            print(f"Call LLM error : {e}")
            return "Sorry could not generate response at this time. Try again"