from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure the Gemini AI SDK with your API key
genai.configure(api_key="your_api_key")  # Replace with your actual Gemini API key

# Define the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_question', methods=['POST'])
def generate_question():
    topic = request.json.get('topic')
    prompt = f"Generate a question about {topic}."
    
    try:
        # Start a chat session and send the prompt
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        question = response.text
        return jsonify({'question': question})
    except Exception as e:
        return jsonify({'error': f'Failed to generate question: {str(e)}'})

@app.route('/validate_answer', methods=['POST'])
def validate_answer():
    question = request.json.get('question')
    user_answer = request.json.get('answer')
    prompt = f"Question: {question}\nAnswer: {user_answer}\nEvaluate the correctness and relevance of this answer."
    
    try:
        # Start a chat session and send the prompt
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        evaluation = response.text
        return jsonify({'evaluation': evaluation})
    except Exception as e:
        return jsonify({'error': f'Failed to validate answer: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
