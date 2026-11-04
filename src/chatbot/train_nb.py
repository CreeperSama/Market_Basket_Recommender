import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

PREPROCESSED_DATA_PATH = "C:/Users/natha/Downloads/Product Recommendation System/data/processed/nlp_training_data.csv"
MODEL_PATH = "C:/Users/natha/Downloads/Product Recommendation System/models/naive_bayes_intent.pkl"
VECTORIZER_PATH = "C:/Users/natha/Downloads/Product Recommendation System/models/tf_idf_vectorizer.pkl"


def main():
    print("Loading the preprocessed data and the vectorizer...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Removing the NaN values from the dataset: ")
    df.dropna(subset=['clean_text'], inplace=True)

    print("Transforming the text into Vectors: ")
    X = vectorizer.transform(df['clean_text'])
    y = df['intent']

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
    )

    nb_classifier = MultinomialNB(alpha=1.0)
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)

    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(f"Classification Report: {classification_report(y_test, y_pred)}")

    print("Saving the trained model: ")
    joblib.dump(nb_classifier, MODEL_PATH)
    print(f"Success ! Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
