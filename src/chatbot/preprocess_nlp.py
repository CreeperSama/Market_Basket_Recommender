import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

RAW_DATA_PATH = "C:/Users/natha/Downloads/Product Recommendation System/data/raw/chatbot/intents.csv.csv"
PROCESSED_DATA_PATH = "C:/Users/natha/Downloads/Product Recommendation System/data/processed/nlp_training_data.csv"
VECTORIZER_PATH = "C:/Users/natha/Downloads/Product Recommendation System/models/tf_idf_vectorizer.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    df = df[["instruction", "intent"]].copy()

    df.rename(columns={'instruction':'text'}, inplace = True)

    df['clean_text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(df['clean_text'])

    joblib.dump(vectorizer, VECTORIZER_PATH)
    df.to_csv(PROCESSED_DATA_PATH, index = False)

    print(f"Success! Vectorizer saved to {VECTORIZER_PATH}")
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()



