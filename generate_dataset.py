import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(42)

COURSE_META_PATH = "large_course_dataset_3000.csv"
OUTPUT_PATH = "course_ratings_large.csv"

def main():
    print("Loading metadata...")
    df_courses = pd.read_csv(COURSE_META_PATH)
    print("Loaded", len(df_courses), "courses.")

    course_ids = df_courses["course_id"].tolist()

    users = [f"U{i}" for i in range(1, 501)]
    rows = []

    print("Generating user ratings...")
    for user in users:
        num_ratings = np.random.randint(50, 120)  # each user rates 50–120 courses
        rated_courses = np.random.choice(course_ids, num_ratings, replace=False)

        for cid in rated_courses:
            base_rating = df_courses.loc[df_courses["course_id"] == cid, "rating"].iloc[0]
            noise = np.random.normal(0, 0.4)
            rating = max(1.0, min(5.0, base_rating + noise))
            rows.append([user, cid, round(rating, 1)])

    df = pd.DataFrame(rows, columns=["user_id", "course_id", "rating"])
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved", len(df), "ratings →", OUTPUT_PATH)
    print("DONE.")

if __name__ == "__main__":
    main()
