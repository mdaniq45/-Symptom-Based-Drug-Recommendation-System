import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st
import csv

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load dataset
file_path = r"C:\Users\hi\OneDrive\Desktop\data science project\NLP PROJECTS\FDA.csv"

df = pd.read_csv(file_path, encoding="ISO-8859-1", quoting=csv.QUOTE_ALL, on_bad_lines="skip")

# Filter dataset to remove non-medicine entries
df_filtered = df[['reason_for_recall', 'product_description']].dropna().copy()

# Remove irrelevant product descriptions (non-drugs)
exclude_keywords = ['bulk', 'capsules', 'extract', 'powder', 'dietary', 'supplement', 
                    'liquid', 'herbal', 'natural', 'essential', 'bark', 'root', 'vitamin', 'oil']

df_filtered = df_filtered[~df_filtered['product_description']
                          .str.contains('|'.join(exclude_keywords), case=False, regex=True)]

# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply text cleaning
df_filtered['clean_reason'] = df_filtered['reason_for_recall'].apply(clean_text)

# Convert text to numerical form using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_filtered['clean_reason'])
y = df_filtered['product_description']

# Split dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Drug Recommendation Function
def recommend_drug(symptom, model, vectorizer, df):
    symptom_clean = clean_text(symptom)
    symptom_vec = vectorizer.transform([symptom_clean])  # Convert input symptom into TF-IDF vector
    predicted_drug = model.predict(symptom_vec)[0]  # Get the predicted drug

    # Ensure the predicted drug is valid
    if predicted_drug in df['product_description'].values:
        return predicted_drug
    return "No suitable medicine found."

# Streamlit App
st.title("💊 Symptom-Based Drug Recommendation System")

# User Input
user_symptom = st.text_input("Enter your symptom:")

if st.button("Recommend Drug"):
    if user_symptom:
        result = recommend_drug(user_symptom, model, vectorizer, df_filtered)
        st.write(f"**Recommended Drug:** {result}")
    else:
        st.write("Please enter a symptom to get recommendations.")
