
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Data
ratings = pd.read_csv("course_ratings_large.csv")
courses = pd.read_csv("large_course_dataset_3000.csv")
P, Q, user_to_index, item_to_index = joblib.load("course_recommender.joblib")

# PREDEFINED TOPICS & BOOKS
category_info = {
    "AI": {
        "topics": ["Neural Networks", "Supervised Learning", "Unsupervised Learning",
                   "Reinforcement Learning Basics", "Evaluation Metrics"],
        "book": "Artificial Intelligence: A Modern Approach – Stuart Russell & Peter Norvig"
    },
    "Data Science": {
        "topics": ["Data Cleaning", "EDA", "Feature Engineering",
                   "Model Selection", "Visualization"],
        "book": "Hands-On Data Science with Python – Packt Publishing"
    },
    "Programming": {
        "topics": ["OOP Concepts", "Data Structures", "Algorithms",
                   "Debugging Techniques", "File Handling"],
        "book": "Automate the Boring Stuff with Python – Al Sweigart"
    },
    "Web Development": {
        "topics": ["HTML & CSS", "JavaScript Basics", "APIs", "SQL Databases", "Deployment"],
        "book": "Full Stack Web Development with Django & React – Packt"
    },
    "Cybersecurity": {
        "topics": ["Network Security", "Ethical Hacking Basics", "Cryptography",
                   "Vulnerability Analysis", "Penetration Testing"],
        "book": "The Web Application Hacker’s Handbook – Dafydd Stuttard"
    },
    "Cloud": {
        "topics": ["IAM", "EC2", "S3", "VPC", "Lambda"],
        "book": "AWS Certified Cloud Practitioner – Official Study Guide"
    },
    "Blockchain": {
        "topics": ["Smart Contracts", "Ethereum Basics", "DApps",
                   "Consensus Algorithms", "Crypto Wallet Security"],
        "book": "Mastering Blockchain – Imran Bashir"
    }
}

# Recommendation Function
def get_recommendations(user_id, top_n=5):
    if user_id not in user_to_index:
        return None

    u_idx = user_to_index[user_id]
    scores = np.dot(P[u_idx], Q.T)

    user_history = ratings[ratings["user_id"] == user_id]["course_id"].tolist()
    course_ids = list(item_to_index.keys())

    scored = []
    for cid, score in zip(course_ids, scores):
        if cid not in user_history:
            scored.append((cid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_courses = scored[:top_n]

    final = []
    for cid, sc in top_courses:
        row = courses[courses["course_id"] == cid].iloc[0]
        final.append({
            "course_id": cid,
            "course_name": row["course_name"],
            "category": row["category"]
        })

    return final

# Get User Interests
def get_user_interests(user_id):
    user_data = ratings[ratings["user_id"] == user_id]
    merged = user_data.merge(courses, on="course_id")
    return merged["category"].value_counts().head(3).index.tolist()

# ------------------------- STREAMLIT UI -------------------------

st.title("📘 Course Recommendation System")

user_list = sorted(ratings["user_id"].unique())
user_id = st.selectbox("Choose User ID", user_list)

if st.button("Get Recommendations"):
    recs = get_recommendations(user_id)
    interests = get_user_interests(user_id)

    st.subheader(f"🎯 Interested Fields for {user_id}:")
    st.write(", ".join(interests))

    st.subheader(f"⭐ Top 5 Recommendations for {user_id}:")

    for i, rec in enumerate(recs, start=1):
        cat = rec["category"]
        topics = category_info.get(cat, {}).get("topics", [])
        book = category_info.get(cat, {}).get("book", "No book available")

        st.write(f"### 📘 Recommended Course {i}")
        st.write(f"**Course ID:** {rec['course_id']}")
        st.write(f"**Course Name:** {rec['course_name']}")
        st.write(f"**Category:** {rec['category']}")

        st.write("**🔑 Topics to Learn:**")
        for t in topics[:5]:
            st.write(f"- {t}")

        st.write(f"**📖 Recommended Book:** {book}")
        st.write("---")
