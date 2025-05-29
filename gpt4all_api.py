from flask import Flask, request, jsonify
from gpt4all import GPT4All

app = Flask(__name__)
model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")

@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    with model.chat_session():
        response = model.generate(prompt, temp=0.7)
        return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)