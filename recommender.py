
import joblib
import numpy as np
import pandas as pd

def load_model():
    return joblib.load("course_recommender.joblib")

def get_recommendations(user_id, top_k=5):
    P, Q, user_to_index, item_to_index = load_model()

    if user_id not in user_to_index:
        raise ValueError("User ID not found!")

    user_idx = user_to_index[user_id]
    scores = np.dot(P[user_idx], Q.T)

    ratings_df = pd.read_csv("course_ratings_large.csv")
    rated_items = ratings_df[ratings_df["user_id"] == user_id]["course_id"].tolist()

    for item in rated_items:
        scores[item_to_index[item]] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]

    items = list(item_to_index.keys())

    return [(items[i], float(scores[i])) for i in top_indices]


if __name__ == "__main__":
    print("Top recommendations for user U1:")
    recs = get_recommendations("U1", top_k=5)
    for cid, score in recs:
        print(f"{cid}: {score:.4f}")
