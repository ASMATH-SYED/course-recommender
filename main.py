
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# ---------------------------
# MATRIX FACTORIZATION MODEL
# ---------------------------

class MatrixFactorization:
    def __init__(self, R, K, learning_rate, reg_param, epochs):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.lr = learning_rate
        self.reg = reg_param
        self.epochs = epochs

        # Initialize latent matrices
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

    def train(self):
        for epoch in range(self.epochs):
            loss = 0
            for u in range(self.num_users):
                for i in range(self.num_items):
                    if self.R[u, i] > 0:
                        prediction = np.dot(self.P[u, :], self.Q[i, :].T)
                        error = self.R[u, i] - prediction

                        loss += error ** 2

                        # Gradient descent updates
                        self.P[u, :] += self.lr * (error * self.Q[i, :] - self.reg * self.P[u, :])
                        self.Q[i, :] += self.lr * (error * self.P[u, :] - self.reg * self.Q[i, :])

            rmse = np.sqrt(loss / np.count_nonzero(self.R))
            print(f"Epoch {epoch + 1}/{self.epochs} - RMSE: {rmse:.4f}")

        return self.P, self.Q


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except:
        raise FileNotFoundError(f"Dataset not found at {path}. Run generate_dataset.py first.")


def save_model(model_data, filename="course_recommender.joblib"):
    joblib.dump(model_data, filename)
    print(f"Saved trained course recommender to {filename}")


# ---------------------------
# MAIN PROCESS
# ---------------------------

def main():
    DATA_PATH = "course_ratings_large.csv"

    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Building matrices...")

    users = sorted(df["user_id"].unique())
    items = sorted(df["course_id"].unique())

    user_to_index = {u: i for i, u in enumerate(users)}
    item_to_index = {c: i for i, c in enumerate(items)}

    R = np.zeros((len(users), len(items)))

    for row in df.itertuples():
        R[user_to_index[row.user_id], item_to_index[row.course_id]] = row.rating

    print("Training Matrix Factorization model...")

    model = MatrixFactorization(
        R,
        K=50,
        learning_rate=0.005,
        reg_param=0.02,
        epochs=20
    )

    P, Q = model.train()

    print("Evaluating model...")

    predictions = np.dot(P, Q.T)
    mask = R > 0
    rmse = np.sqrt(np.sum((predictions[mask] - R[mask]) ** 2) / np.count_nonzero(mask))

    print(f"Test RMSE: {rmse:.4f}")

    print("Saving model...")
    save_model((P, Q, user_to_index, item_to_index))

    print("Training complete!")


if __name__ == "__main__":
    main()
