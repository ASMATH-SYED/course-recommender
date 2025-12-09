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

rating_counts = ratings["course_id"].value_counts()
popularity = (rating_counts / rating_counts.max()).to_dict()


# -------------------------------------------------
# FIELD INFO (Topics + Accurate Books)
# -------------------------------------------------
field_info = {
    "Artificial Intelligence (AI)": {
        "topics": ["Search & Problem Solving", "Knowledge Representation", "Inference", "Decision Making", "AI Ethics"],
        "book": "Artificial Intelligence: A Modern Approach – Stuart Russell & Peter Norvig"
    },
    "Machine Learning (ML)": {
        "topics": ["Supervised Learning", "Unsupervised Learning", "Feature Engineering", "Model Evaluation", "Hyperparameter Tuning"],
        "book": "Hands-On Machine Learning – Aurélien Géron"
    },
    "Deep Learning": {
        "topics": ["Neural Networks", "CNN", "RNN/LSTM", "Transfer Learning", "Regularization"],
        "book": "Deep Learning with Python – François Chollet"
    },
    "Data Science (DS)": {
        "topics": ["Data Cleaning", "EDA", "Model Building", "Feature Engineering", "Visualization"],
        "book": "Data Science for Beginners – Andrew Park"
    },
    "Programming Fundamentals": {
        "topics": ["Variables", "Loops", "Functions", "Data Structures", "OOP Basics"],
        "book": "Learning Python – Mark Lutz"
    },
    "Cybersecurity": {
        "topics": ["Network Basics", "Scanning", "Exploitation", "Web Security", "Pen Testing"],
        "book": "The Web Application Hacker’s Handbook – Dafydd Stuttard"
    },
    "Cloud Computing": {
        "topics": ["IAM", "EC2", "S3", "VPC", "Serverless (Lambda)"],
        "book": "AWS Certified Cloud Practitioner – Official Guide"
    },
    "Web Development": {
        "topics": ["HTML/CSS", "JavaScript", "Backend (Django/Node)", "APIs", "Deployment"],
        "book": "Eloquent JavaScript – Marijn Haverbeke"
    },
    "Mobile App Development": {
        "topics": ["Android/iOS Basics", "UI Design", "API Integration", "State Management", "Publishing"],
        "book": "Android Programming: The Big Nerd Ranch Guide"
    },
    "Internet of Things (IoT)": {
        "topics": ["Sensors", "Microcontrollers", "Protocols", "IoT Cloud", "IoT Security"],
        "book": "Internet of Things: A Hands-On Approach – Arshdeep Bahga"
    },
    "Blockchain": {
        "topics": ["Blockchain Basics", "Smart Contracts", "Ethereum", "Consensus Algorithms", "Web3"],
        "book": "Mastering Blockchain – Imran Bashir"
    },
    "Software Engineering": {
        "topics": ["SDLC", "Clean Code", "Testing", "Git", "System Design"],
        "book": "Clean Code – Robert C. Martin"
    },
    "DevOps": {
        "topics": ["CI/CD", "Git", "Docker", "Kubernetes", "Monitoring"],
        "book": "The DevOps Handbook – Gene Kim"
    },
    "Database Systems": {
        "topics": ["SQL Queries", "Joins", "Indexing", "Transactions", "NoSQL"],
        "book": "Database System Concepts – Silberschatz"
    },
    "Business Analytics": {
        "topics": ["KPIs", "Dashboards", "Reporting", "Predictive Analysis", "Visualization"],
        "book": "Data Science for Business – Provost & Fawcett"
    },
    "Product Management": {
        "topics": ["User Research", "MVP", "Roadmapping", "Metrics", "Team Communication"],
        "book": "Inspired – Marty Cagan"
    },
    "UI/UX Design": {
        "topics": ["User Research", "Wireframing", "Prototyping", "Accessibility", "Usability Testing"],
        "book": "The Design of Everyday Things – Don Norman"
    },
    "Digital Marketing": {
        "topics": ["SEO", "Social Media", "Content Strategy", "Email Marketing", "Analytics"],
        "book": "Digital Marketing for Dummies – Ryan Deiss"
    },
    "Project Management": {
        "topics": ["Planning", "Agile", "Scrum", "Risk Management", "Tracking"],
        "book": "Scrum – Jeff Sutherland"
    }
}

DEFAULT_FIELD = "Programming Fundamentals"


# -------------------------------------------------
# FIELD MAPPING (Strong, Correct, No mismatches)
# -------------------------------------------------
def infer_field(row):
    name = str(row["course_name"]).lower()

    mapping = {
        "computer vision": "Deep Learning",
        "deep learning": "Deep Learning",
        "machine learning": "Machine Learning (ML)",
        "mlops": "Machine Learning (ML)",
        "artificial intelligence": "Artificial Intelligence (AI)",
        "ai": "Artificial Intelligence (AI)",
        "data science": "Data Science (DS)",
        "python for data science": "Data Science (DS)",
        "python": "Programming Fundamentals",
        "java": "Programming Fundamentals",
        "c++": "Programming Fundamentals",
        "programming": "Programming Fundamentals",
        "ethical hacking": "Cybersecurity",
        "cyber": "Cybersecurity",
        "penetration testing": "Cybersecurity",
        "cloud": "Cloud Computing",
        "aws": "Cloud Computing",
        "azure": "Cloud Computing",
        "web development": "Web Development",
        "frontend": "Web Development",
        "backend": "Web Development",
        "django": "Web Development",
        "react": "Web Development",
        "devops": "DevOps",
        "docker": "DevOps",
        "kubernetes": "DevOps",
        "sql": "Database Systems",
        "database": "Database Systems",
        "power bi": "Business Analytics",
        "tableau": "Business Analytics",
        "business analytics": "Business Analytics",
        "android": "Mobile App Development",
        "ios": "Mobile App Development",
        "flutter": "Mobile App Development",
        "iot": "Internet of Things (IoT)",
        "internet of things": "Internet of Things (IoT)",
        "blockchain": "Blockchain",
        "web3": "Blockchain",
        "smart contract": "Blockchain",
        "product management": "Product Management",
        "ui": "UI/UX Design",
        "ux": "UI/UX Design",
        "marketing": "Digital Marketing",
        "project management": "Project Management",
        "agile": "Project Management",
        "scrum": "Project Management",
    }

    for key, value in mapping.items():
        if key in name:
            return value

    return DEFAULT_FIELD


# Apply mapping
courses["field"] = courses.apply(infer_field, axis=1)


# -------------------------------------------------
# USER PROFILE
# -------------------------------------------------
def get_user_profile_fields(user_id):
    user_rows = ratings[ratings["user_id"] == user_id]
    merged = user_rows.merge(courses, on="course_id")

    if merged.empty:
        return []

    return list(merged["field"].value_counts().index)


# -------------------------------------------------
# RECOMMENDER ENGINE
# -------------------------------------------------
def get_recommendations(user_id, top_n, profile_fields):
    if user_id not in user_to_index:
        return []

    u_idx = user_to_index[user_id]

    scores = np.dot(P[u_idx], Q.T)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    user_history = set(ratings[ratings["user_id"] == user_id]["course_id"])
    recommendations = []

    for cid, score in zip(item_to_index.keys(), scores):
        if cid in user_history:
            continue

        row = courses[courses["course_id"] == cid].iloc[0]
        field = row["field"]

        content_score = 1 if field in profile_fields else 0.5
        final_score = 0.6 * score + 0.25 * popularity.get(cid, 0) + 0.15 * content_score

        recommendations.append({
            "course_id": cid,
            "course_name": row["course_name"],
            "field": field,
            "final_score": final_score
        })

    recommendations.sort(key=lambda x: x["final_score"], reverse=True)

    final, used = [], set()
    for r in recommendations:
        if r["field"] not in used:
            used.add(r["field"])
            final.append(r)
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
    st.markdown("<h4 style='text-align:center;'>By <b>SYED ASMATH</b></h4>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    user = st.selectbox("Choose User ID", sorted(ratings["user_id"].unique()))
    num = st.slider("Number of recommendations", 1, 5, 5)

    if st.button("Get Recommendations"):
        fields = get_user_profile_fields(user)
        recs = get_recommendations(user, num, fields)

        if recs:
            st.subheader(f"🎯 Interested Fields for {user}:")
            st.write(", ".join({r["field"] for r in recs}))

            st.subheader(f"⭐ Top {len(recs)} Recommendations for {user}:")

            for i, rec in enumerate(recs, start=1):
                info = field_info[rec["field"]]

                st.markdown(f"<h3>📘 Recommended Course {i}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Course ID:</b> {rec['course_id']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Course Name:</b> {rec['course_name']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Field:</b> {rec['field']}</p>", unsafe_allow_html=True)

                st.markdown("<p><b>🔑 Topics to Learn:</b></p>", unsafe_allow_html=True)
                for t in info["topics"]:
                    st.markdown(f"<p>- {t}</p>", unsafe_allow_html=True)

                st.markdown(f"<p><b>📖 Recommended Book:</b> {info['book']}</p>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

        else:
            st.warning("No recommendations available.")