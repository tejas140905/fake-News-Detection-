from flask import Flask, request, jsonify
import pickle
from src.data_preprocessing import clean_text

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('model/fake_news_model.pkl', 'rb'))
tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the POST request
    text = request.form['text']
    
    # Preprocess the text
    text = clean_text(text)
    
    # Transform the text using the same TfidfVectorizer
    text_tfidf = tfidf.transform([text])
    
    # Predict the label (0: fake, 1: real)
    prediction = model.predict(text_tfidf)
    
    # Return the prediction result
    return jsonify({'prediction': 'Real' if prediction[0] == 1 else 'Fake'})

if __name__ == '__main__':
    app.run(debug=True)
