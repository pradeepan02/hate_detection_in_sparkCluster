from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources (ensure these are downloaded only once)
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and the TF-IDF vectorizer
with open('gbtmodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Preprocessing function for input text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs and special characters
    text = re.sub(r"https?\S+|www\S+|http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return " ".join(filtered_tokens)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get tweet from the form
    text = request.form['tweet']

    # Preprocess the input text
    cleaned_text = preprocess_text(text)

    # Transform the cleaned text using the loaded TF-IDF vectorizer
    transformed_features = tfidf_vectorizer.transform([cleaned_text]).toarray()

    # Make the prediction using the loaded model
    prediction = model.predict(transformed_features)

    # Interpret the result
    result = "Positive Speech" if prediction[0] == 0 else "Offensive Speech"
    
    # Render the result in the HTML page
    return render_template('index.html', prediction_text=f"Predicted Sentiment: {result}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
