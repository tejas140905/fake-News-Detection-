import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean the text data
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove special chars
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load the dataset
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Remove rows with missing values
    df['cleaned_text'] = df['text'].apply(clean_text)  # Apply text cleaning function
    return df

# Split data into training and testing sets
def split_data(df):
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# TF-IDF Vectorization
def vectorize_data(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)  # Fit and transform training data
    X_test_tfidf = tfidf.transform(X_test)  # Transform test data
    return tfidf, X_train_tfidf, X_test_tfidf
