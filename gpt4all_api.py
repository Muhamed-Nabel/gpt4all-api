from flask import Flask, request, jsonify
from gpt4all import GPT4All
import os

app = Flask(__name__)

model_name = os.getenv("MODEL_NAME", "mistral-7b-instruct-v0.1.Q4_0.gguf")
model = GPT4All(model_name)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = data["prompt"]
    with model.chat_session():
        response = model.generate(prompt, temp=0.7)
        return jsonify({"response": response})

@app.route('/', methods=['GET'])
def index():
    return "Hello, GPT4All API is running. Use POST /chat with JSON {\"prompt\": \"...\"}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 10000)))
