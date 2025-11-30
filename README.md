# Course Recommendation System using Collaborative Filtering

## Overview
This is a minimal, ready-to-run reference implementation for the project
**"Course Recommendation System using Collaborative Filtering"** suitable for an M.Tech CSE (Data Science) dissertation or mini-project.

The pipeline is:
1. Load data from a CSV file.
2. Clean / preprocess features.
3. Split into train / test sets.
4. Train baseline models.
5. Evaluate and save metrics.
6. Persist the trained model with `joblib` for later reuse.

## How to Run

```bash
pip install -r ../requirements.txt
python main.py
```

Make sure to place your dataset as `data.csv` inside this folder,
or update the `DATA_PATH` variable in `main.py`.
