import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# LOAD DATA & MODEL
# -------------------------------------------------
ratings = pd.read_csv("course_ratings_large.csv")
courses = pd.read_csv("large_course_dataset_3000.csv")
P, Q, user_to_index, item_to_index = joblib.load("course_recommender.joblib")

# Popularity scores
rating_counts = ratings["course_id"].value_counts()
popularity = (rating_counts / rating_counts.max()).to_dict()

# -------------------------------------------------
# FIELD INFO
# -------------------------------------------------
field_info = {
    "Artificial Intelligence (AI)": {
        "topics": ["Search", "Knowledge Representation", "Inference", "Decision Making", "Ethics"],
        "book": "Artificial Intelligence: A Modern Approach – Russell & Norvig"
    },
    "Machine Learning (ML)": {
        "topics": ["Supervised", "Unsupervised", "Feature Engineering", "Evaluation", "Tuning"],
        "book": "Hands-On Machine Learning – Aurélien Géron"
    },
    "Deep Learning": {
        "topics": ["Neural Networks", "CNN", "RNN/LSTM", "Transfer Learning", "Regularization"],
        "book": "Deep Learning with Python – François Chollet"
    },
    "Data Science (DS)": {
        "topics": ["Data Cleaning", "EDA", "Modeling", "Feature Engineering", "Visualization"],
        "book": "Python for Data Analysis – Wes McKinney"
    },
    "Cloud Computing": {
        "topics": ["IAM", "EC2", "S3", "VPC", "Serverless"],
        "book": "AWS Certified Cloud Practitioner – Official Guide"
    },
    "Cybersecurity": {
        "topics": ["Networking", "Scanning", "Exploitation", "Web Security", "Pen Testing"],
        "book": "The Web Application Hacker’s Handbook – Dafydd Stuttard"
    },
    "Web Development": {
        "topics": ["HTML", "CSS", "JavaScript", "Backend", "Deployment"],
        "book": "Eloquent JavaScript – Marijn Haverbeke"
    },
    "Business Analytics": {
        "topics": ["KPIs", "Dashboards", "Reporting", "Insights", "Visualization"],
        "book": "Data Science for Business – Provost & Fawcett"
    },
    "Programming Fundamentals": {
        "topics": ["Variables", "Loops", "Functions", "OOP", "Data Structures"],
        "book": "Introduction to Programming Using Python – John Zelle"
    }
}

DEFAULT_FIELD = "Programming Fundamentals"

# -------------------------------------------------
# FIELD INFERENCE
# -------------------------------------------------
def infer_field(row):
    name = str(row["course_name"]).lower()

    rules = {
        "machine learning": "Machine Learning (ML)",
        "deep learning": "Deep Learning",
        "computer vision": "Deep Learning",
        "artificial intelligence": "Artificial Intelligence (AI)",
        "data science": "Data Science (DS)",
        "cloud": "Cloud Computing",
        "aws": "Cloud Computing",
        "cyber": "Cybersecurity",
        "ethical hacking": "Cybersecurity",
        "web development": "Web Development",
        "django": "Web Development",
        "react": "Web Development",
        "business analytics": "Business Analytics",
        "power bi": "Business Analytics",
        "tableau": "Business Analytics",
        "python": "Programming Fundamentals",
        "java": "Programming Fundamentals",
    }

    for k, v in rules.items():
        if k in name:
            return v

    return DEFAULT_FIELD

courses["field"] = courses.apply(infer_field, axis=1)

# -------------------------------------------------
# EXISTING USER PROFILE
# -------------------------------------------------
def get_user_profile_fields(user_id):
    user_rows = ratings[ratings["user_id"] == user_id]
    merged = user_rows.merge(courses, on="course_id")
    if merged.empty:
        return []
    return list(merged["field"].value_counts().index)

# -------------------------------------------------
# EXISTING USER RECOMMENDATION
# -------------------------------------------------
def get_recommendations(user_id, top_n, profile_fields):
    u_idx = user_to_index[user_id]
    scores = np.dot(P[u_idx], Q.T)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    history = set(ratings[ratings["user_id"] == user_id]["course_id"])
    candidates = []

    for cid, score in zip(item_to_index.keys(), scores):
        if cid in history:
            continue

        row = courses[courses["course_id"] == cid].iloc[0]
        field = row["field"]

        content = 1.0 if field in profile_fields else 0.5
        final = 0.6 * score + 0.25 * popularity.get(cid, 0) + 0.15 * content

        candidates.append({
            "course_id": cid,
            "course_name": row["course_name"],
            "field": field,
            "score": final
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    final, used = [], set()
    for c in candidates:
        if c["field"] not in used:
            final.append(c)
            used.add(c["field"])
        if len(final) == top_n:
            break

    return final

# -------------------------------------------------
# NEW USER / GUEST RECOMMENDATION
# -------------------------------------------------
def get_recommendations_for_guest(selected_fields, top_n):
    candidates = []

    for _, row in courses.iterrows():
        if row["field"] in selected_fields:
            score = popularity.get(row["course_id"], 0)
            candidates.append({
                "course_id": row["course_id"],
                "course_name": row["course_name"],
                "field": row["field"],
                "score": score
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    final, used = [], set()
    for c in candidates:
        if c["field"] not in used:
            final.append(c)
            used.add(c["field"])
        if len(final) == top_n:
            break

    return final

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="Course Recommendation System", layout="wide")
left, mid, right = st.columns([1, 2, 1])

with mid:
    st.markdown("<h1 style='text-align:center;'>📘 Course Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>By SYED ASMATH</h4>", unsafe_allow_html=True)
    st.markdown("---")

    user_type = st.radio("Select User Type", ["Existing User", "New User (Guest)"])

    if user_type == "Existing User":
        user = st.selectbox("Choose User ID", sorted(ratings["user_id"].unique()))
    else:
        selected_fields = st.multiselect(
            "Select your interested fields",
            list(field_info.keys())
        )

    num = st.slider("Number of recommendations", 1, 5, 5)

    if st.button("Get Recommendations"):

        if user_type == "Existing User":
            fields = get_user_profile_fields(user)
            recs = get_recommendations(user, num, fields)
        else:
            if not selected_fields:
                st.warning("Please select at least one field")
                st.stop()
            recs = get_recommendations_for_guest(selected_fields, num)

        st.subheader("⭐ Recommended Courses")

        for i, r in enumerate(recs, 1):
            info = field_info[r["field"]]
            st.markdown(f"### 📘 Course {i}")
            st.write(f"Course ID: {r['course_id']}")
            st.write(f"Course Name: {r['course_name']}")
            st.write(f"Field: {r['field']}")
            st.write("Topics to Learn:")
            for t in info["topics"]:
                st.write(f"- {t}")
            st.write(f"Recommended Book: {info['book']}")
            st.markdown("---")
