import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

FEATURE_MATRIX_PATH = "C:/Users/natha/Downloads/Product Recommendation System/data/processed/instacart_feature_matrix.csv"
RF_MODEL_PATH = "C:/Users/natha/Downloads/Product Recommendation System/models/random_forest_recommender.pkl"

def main():
    print("1. Loading the cheat sheets (Feature Matrix)...")
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    df = df.sample(n=200000, random_state=42)
    
    # Separate the answers (target) from the study material (features)
    y = df['target']
    X = df.drop(['user_id', 'product_id', 'target'], axis=1)

    print("2. Hiding some data for a pop quiz...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Studying (Training the Random Forest)...")
    # Create 100 decision trees to vote on the outcomes
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("4. Grading the pop quiz...")
    # Predict the probability of someone buying the item
    y_predictions = rf.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_predictions)
    print(f"Model Grade (ROC-AUC): {score:.2f}")

    print("5. Saving the model's brain...")
    joblib.dump(rf, RF_MODEL_PATH)
    print("Saved to models folder!")

if __name__ == "__main__":
    main()