from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from src.data_preprocessing import load_and_clean_data, split_data, vectorize_data

# Load dataset
df = load_and_clean_data('data/fake_news_data.csv')

# Split data into train and test sets
X_train, X_test, y_train, y_test = split_data(df)

# Vectorize the data using TF-IDF
tfidf, X_train_tfidf, X_test_tfidf = vectorize_data(X_train, X_test)

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer for later use
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
