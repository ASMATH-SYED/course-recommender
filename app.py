import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# LOAD DATA & MODEL
# -------------------------
ratings = pd.read_csv("course_ratings_large.csv")
courses = pd.read_csv("large_course_dataset_3000.csv")
P, Q, user_to_index, item_to_index = joblib.load("course_recommender.joblib")

# -------------------------
# COURSE → TOPICS + BOOK MAPPING
# -------------------------
course_info = {
    "Python for Data Science": {
        "topics": ["Python Basics", "NumPy", "Pandas", "Data Cleaning", "Visualization"],
        "book": "Python for Data Analysis – Wes McKinney",
        "field": "Data Science"
    },
    "Full Stack Web Development": {
        "topics": ["HTML & CSS", "JavaScript", "Backend (Django/Node)", "REST APIs", "Database & Deployment"],
        "book": "Eloquent JavaScript – Marijn Haverbeke",
        "field": "Web Development"
    },
    "Cloud Computing with AWS": {
        "topics": ["IAM", "EC2", "S3", "VPC", "Lambda"],
        "book": "AWS Certified Cloud Practitioner – Official Study Guide",
        "field": "Cloud"
    },
    "Deep Learning with TensorFlow": {
        "topics": ["Neural Networks", "Backpropagation", "CNN", "RNN/LSTM", "Dropout & Regularization"],
        "book": "Deep Learning with Python – François Chollet",
        "field": "AI / Machine Learning"
    },
    "Ethical Hacking Masterclass": {
        "topics": ["Network Basics", "Scanning", "Exploitation", "Web Security", "Pen Testing"],
        "book": "The Web Application Hacker’s Handbook – Dafydd Stuttard",
        "field": "Cybersecurity"
    },
    "Natural Language Processing Basics": {
        "topics": ["Text Cleaning", "Tokenization", "Embeddings", "RNN", "Transformers"],
        "book": "Speech and Language Processing – Jurafsky & Martin",
        "field": "AI / Machine Learning"
    }
}

# -------------------------
# FIND COURSE FIELD
# -------------------------
def get_course_field(course_name):
    if course_name in course_info:
        return course_info[course_name]["field"]
    return None  # NEVER return "Other"

# -------------------------
# RECOMMENDATION SYSTEM
# -------------------------
def get_recommendations(user_id, top_n=5):
    if user_id not in user_to_index:
        return None, None

    u_idx = user_to_index[user_id]
    scores = np.dot(P[u_idx], Q.T)

    user_history = ratings[ratings["user_id"] == user_id]["course_id"].tolist()
    course_ids = list(item_to_index.keys())

    valid_courses = []

    for cid, score in zip(course_ids, scores):
        row = courses[courses["course_id"] == cid].iloc[0]
        cname = row["course_name"]

        if cname in course_info and cname not in user_history:
            valid_courses.append((cid, score))

    valid_courses.sort(key=lambda x: x[1], reverse=True)
    top_courses = valid_courses[:top_n]

    results = []
    fields = set()  # store correct fields

    for cid, sc in top_courses:
        row = courses[courses["course_id"] == cid].iloc[0]
        cname = row["course_name"]
        info = course_info[cname]

        results.append({
            "course_id": cid,
            "course_name": cname,
            "topics": info["topics"],
            "book": info["book"],
            "field": info["field"]
        })

        fields.add(info["field"])  # add field to interested fields list

    return results, list(fields)

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📘 Course Recommendation System")

user_list = sorted(ratings["user_id"].unique())
user_id = st.selectbox("Choose User ID", user_list)

if st.button("Get Recommendations"):
    recs, interests = get_recommendations(user_id)

    # interest fields = only fields from recommended courses
    st.subheader(f"🎯 Interested Fields for {user_id}:")
    st.write(", ".join(interests))

    st.subheader(f"⭐ Top 5 Recommendations for {user_id}:")

    for i, rec in enumerate(recs, start=1):
        st.write(f"### 📘 Recommended Course {i}")
        st.write(f"**Course ID:** {rec['course_id']}")
        st.write(f"**Course Name:** {rec['course_name']}")
        st.write(f"**Field:** {rec['field']}")

        st.write("**🔑 Topics to Learn:**")
        for t in rec["topics"]:
            st.write(f"- {t}")

        st.write(f"**📖 Recommended Book:** {rec['book']}")
        st.write("---")
