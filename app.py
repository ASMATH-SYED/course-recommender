
import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "course_recommender.joblib"
COURSE_META_PATH = "large_course_dataset_3000.csv"
RATINGS_PATH = "course_ratings_large.csv"


def load_data():
    courses = pd.read_csv(COURSE_META_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    return courses, ratings


def load_model():
    P, Q, user_to_index, item_to_index = joblib.load(MODEL_PATH)
    return P, Q, user_to_index, item_to_index


def get_recommendations(user_id: str, top_k: int = 5):
    courses, ratings = load_data()
    P, Q, user_to_index, item_to_index = load_model()

    if user_id not in user_to_index:
        return None, f"User '{user_id}' not found."

    u_idx = user_to_index[user_id]
    scores = np.dot(P[u_idx, :], Q.T)

    rated_courses = ratings[ratings["user_id"] == user_id]["course_id"].unique()
    rated_indices = [item_to_index[c] for c in rated_courses if c in item_to_index]

    mask = np.ones_like(scores, dtype=bool)
    mask[rated_indices] = False
    scores_filtered = np.where(mask, scores, -np.inf)

    top_indices = np.argsort(scores_filtered)[::-1][:top_k]

    inv_item_index = {idx: cid for cid, idx in item_to_index.items()}
    rec_course_ids = [inv_item_index[i] for i in top_indices]

    rec_df = courses[courses["course_id"].isin(rec_course_ids)].copy()
    rec_df["predicted_rating"] = [scores[i] for i in top_indices]

    rec_df = rec_df.sort_values("predicted_rating", ascending=False)

    return rec_df, None


def main():
    st.title("📚 Course Recommendation System")

    courses, ratings = load_data()
    user_ids = sorted(ratings["user_id"].unique())

    st.sidebar.header("User Selection")

    selected_user = st.sidebar.selectbox("Choose User ID", user_ids, index=0)
    top_k = st.sidebar.slider("Number of recommendations", 1, 20, 5)

    if st.sidebar.button("Get Recommendations"):
        recs, error = get_recommendations(selected_user, top_k)

        if error:
            st.error(error)
        else:
            st.subheader(f"Top {top_k} recommendations for {selected_user}:")
            st.dataframe(recs.reset_index(drop=True))


if __name__ == "__main__":
    main()
