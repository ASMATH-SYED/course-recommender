
import pandas as pd
import numpy as np

np.random.seed(42)

course_ids = [f"C{i}" for i in range(1, 3001)]

course_names = [
    "Introduction to Artificial Intelligence",
    "Machine Learning A-Z",
    "Deep Learning with TensorFlow",
    "Natural Language Processing Basics",
    "Computer Vision Crash Course",
    "Python for Data Science",
    "Statistics for Machine Learning",
    "Data Visualization using Power BI",
    "Introduction to SQL",
    "Java Programming Masterclass",
    "Full Stack Web Development",
    "Django Web Development",
    "React for Frontend Development",
    "Cybersecurity Basics",
    "Ethical Hacking Masterclass",
    "Cloud Computing with AWS",
    "DevOps Fundamentals",
    "Blockchain Essentials",
    "Advanced AI Model Deployment",
    "Reinforcement Learning Fundamentals",
]

categories = ["AI","Data Science","Programming","Web Development","Cybersecurity","Cloud","Blockchain"]

sub_categories = {
    "AI": ["Artificial Intelligence","Machine Learning","Deep Learning","NLP","Computer Vision","MLOps","Reinforcement Learning"],
    "Data Science": ["Python","Statistics","Visualization","Power BI","Analytics"],
    "Programming": ["Databases","Java","C++","C","Python Scripting"],
    "Web Development": ["Full Stack","Frontend","Backend","Python Web","React"],
    "Cybersecurity": ["General","Hacking","Network Security","Forensics"],
    "Cloud": ["AWS","Azure","DevOps","GCP"],
    "Blockchain": ["General","Smart Contracts","Web3"],
}

difficulty_levels = ["Beginner","Intermediate","Advanced"]

rows = []

for cid in course_ids:
    cat = np.random.choice(categories)
    sub = np.random.choice(sub_categories[cat])
    name = np.random.choice(course_names)
    difficulty = np.random.choice(difficulty_levels)
    rating = round(np.random.uniform(3.5, 5.0), 1)
    students = np.random.randint(200, 5000)
    desc = f"{name} course covering {sub} in {cat}."

    rows.append([cid, name, cat, sub, difficulty, rating, students, desc])

df = pd.DataFrame(rows, columns=[
    "course_id","course_name","category","sub_category",
    "difficulty","rating","num_students","description"
])

df.to_csv("large_course_dataset_3000.csv", index=False)

print("Saved dataset with 3000 rows to large_course_dataset_3000.csv")
