from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from src.chabot_service import ChatbotServices

chatbot = Blueprint("chatbot", __name__)

chat_service = ChatbotServices()

@chatbot.route("/chat/start", methods=["POST"])
def start_chat():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "User not logged in."
        }), 401

    data = request.get_json()
    # prediction_id = request.form.get("prediction_id")
    prediction_id = data.get("prediction_id")

    if not prediction_id:
        return jsonify({
            "success": False,
            "message": "Prediction ID is required"
        }), 400

    conversation = chat_service.start_conversation(prediction_id, session["user_id"])

    if not conversation["success"]:
        return jsonify(conversation), 400

    conversation_data = chat_service.get_conversation(conversation["conversation_id"], session["user_id"])

    return jsonify({
        "success": True,
        "conversation_id":conversation["conversation_id"],
        "messages" : conversation_data["messages"],
        "remaining_questions" : conversation_data["remaining_questions"]
    })

@chatbot.route("/chat/send", methods=["POST"])
def send_message():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "User not logged in."
        }), 401

    # conversation_id = request.form.get("conversation_id")

    # question = request.form.get("question", "").strip()
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    question = data.get("question", "").strip()

    if not conversation_id:
        return jsonify({
            "success": False,
            "message": "Conversation ID is missing"
        }), 400
    
    if not question:
        return jsonify({
            "success": False,
            "message": "Question cannot be empty"
        }), 400

    response = chat_service.ask_question(conversation_id=conversation_id, user_id=session["user_id"], question=question)

    return jsonify(response)

@chatbot.route("/chat/history/<int:conversation_id>")
def chat_history(conversation_id):

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "User not logged in."
        }), 401

    conversation = chat_service.get_conversation(conversation_id, session["user_id"])

    return jsonify(conversation)

@chatbot.route("/chat/remaining/<int:conversation_id>")
def remaining_questions(conversation_id):

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "User not logged in."
        }), 401

    remaining = chat_service.get_remaining_questions(conversation_id)

    return jsonify({
        "success": True,
        "remaining_questions": remaining
    })