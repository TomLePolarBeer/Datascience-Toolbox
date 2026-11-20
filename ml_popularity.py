"""
ml_popularity.py - Research question 1:
"Which musical characteristics are associated with higher song popularity?"

Approach: regression. A Random Forest predicts popularity (0-100) from the
audio features only, then we read which features the model relies on most.

Run:    python data_exam.py      (creates data/spotify_clean.csv)
        python ml_popularity.py
Output: data/ml_feature_importance.csv  (used by the dashboard)
        data/ml_metrics.csv             (R2 / MAE of each model)
"""
import os

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Relative paths work from any machine / working directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CLEAN_PATH = "data/spotify_clean.csv"
RANDOM_STATE = 42  # fixed so everyone gets the same numbers

# --------------------------------------------------------------------------
# 1. Load the cleaned data (one row per track_id -> no song in both train and test)
# --------------------------------------------------------------------------
df = pd.read_csv(CLEAN_PATH)
df["duration_min"] = df["duration_ms"] / 60000
df["explicit"] = df["explicit"].astype(int)

# --------------------------------------------------------------------------
# 2. Target and features
# --------------------------------------------------------------------------
# Target: popularity (0-100). Tracks with popularity 0 are KEPT: they are real
# tracks that nobody listens to, and the model must see unpopular songs too.
TARGET = "popularity"

# Features: musical characteristics only.
# Excluded on purpose:
#   - tempo: 157 tracks have tempo = 0 (detection error), measure judged unreliable
#   - artists: the same artist is in train and test -> the model would learn
#     "this artist is famous", not the music (leakage)
#   - track_genre: not a musical characteristic, and only one genre per track
#     is left after de-duplication
#   - ids and names (track_id, track_name, album_name): no musical meaning
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "duration_min",
    "mode", "key", "explicit",
]


def evaluate(data, label):
    """Train a baseline and a Random Forest on `data`, return metrics + importances."""
    X = data[FEATURES]
    y = data[TARGET]

    # 80% to learn, 20% to test on songs the model has never seen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Baseline: always predicts the average popularity of the training set.
    # Any real model has to beat this to be worth something.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    # Random Forest: 100 decision trees, the prediction is their average.
    # min_samples_leaf=5 keeps trees from memorising single songs (and saves RAM).
    model = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=2
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "dataset": label,
        "rows": len(data),
        "baseline_r2": r2_score(y_test, y_base),
        "baseline_mae": mean_absolute_error(y_test, y_base),
        "rf_r2": r2_score(y_test, y_pred),
        "rf_mae": mean_absolute_error(y_test, y_pred),
    }
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return metrics, importance


# --------------------------------------------------------------------------
# 3. Main model: all tracks (popularity 0 kept)
# --------------------------------------------------------------------------
metrics_all, importance_all = evaluate(df, "all tracks")

# --------------------------------------------------------------------------
# 4. Robustness check: same model without the popularity-0 tracks
# --------------------------------------------------------------------------
metrics_no0, importance_no0 = evaluate(df[df[TARGET] > 0], "without popularity 0")

# --------------------------------------------------------------------------
# 5. Results
# --------------------------------------------------------------------------
metrics = pd.DataFrame([metrics_all, metrics_no0])
pd.set_option("display.width", 120)
print("=" * 70)
print("MODEL PERFORMANCE (test set, 20%)")
print("=" * 70)
print(metrics.round(3).to_string(index=False))

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE - main model (all tracks)")
print("=" * 70)
print(importance_all.round(3).to_string(index=False))

# Direction of the link: importance says HOW MUCH a feature matters, not
# whether it goes up or down with popularity. The correlation gives the sign.
print("\nDirection (correlation with popularity):")
print(df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).round(3)
      .reindex(importance_all["feature"]).to_string())

# Saved for the Dash dashboard (ML indicator)
importance_all.to_csv("data/ml_feature_importance.csv", index=False)
metrics.to_csv("data/ml_metrics.csv", index=False)
print("\nSaved data/ml_feature_importance.csv and data/ml_metrics.csv")
