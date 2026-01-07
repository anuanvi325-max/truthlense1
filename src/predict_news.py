import pickle
import re

with open('../models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(news_text):
    clean = clean_text(news_text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    return prediction

if __name__ == "__main__":
    news = input("Enter news headline/content: ")
    result = predict_news(news)
    print(f"\nPrediction: {result.upper()}")
