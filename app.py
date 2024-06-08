from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import re

app = Flask(__name__)

# Load the model, tokenizer, and parameters
model = load_model('sarcasm_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('params.json', 'r') as json_file:
    params = json.load(json_file)
    max_length = params['max_length']
    padding_type = params['padding_type']
    trunc_type = params['trunc_type']

def get_title(url):
    match = re.search(r'\/([^\/]+)\/?$', url)
    if match:
        title_with_dashes = match.group(1)
        title_with_spaces = title_with_dashes.replace("-", " ")
        return title_with_spaces
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    title = get_title(url)
    if title is None:
        return jsonify({'error': 'Title not found in the provided URL'}), 400

    # Preprocess the title
    example_text = [title]
    example_seq = tokenizer.texts_to_sequences(example_text)
    example_padded = pad_sequences(example_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Make prediction
    prediction = model.predict(example_padded)
    score = float(prediction[0][0])  # Convert to native Python float

    return jsonify({
        'title': title,
        'score' : score,
        'sarcastic': bool(score > 0.5)
    })

if __name__ == '__main__':
    app.run(debug=True)
