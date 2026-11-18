"""
data_exam.py - examine and clean the Spotify tracks dataset.

Run from anywhere:  python data_exam.py
Input : data/spotify-tracks-dataset.csv   (raw file, never modified)
Output: data/spotify_clean.csv            (one row per track_id)
"""
import os

import pandas as pd

# Make the relative paths below work from any working directory (any machine).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = "data/spotify-tracks-dataset.csv"
CLEAN_PATH = "data/spotify_clean.csv"

# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

# --------------------------------------------------------------------------
# 2. Examine the raw data
# --------------------------------------------------------------------------
print("=" * 70)
print("RAW DATA")
print("=" * 70)
print(f"Shape: {df.shape}\n")

print("Columns and types:")
print(df.dtypes, "\n")

print("Min / max of numeric columns:")
print(df.select_dtypes(include="number").agg(["min", "max"]).T, "\n")

print("Unique values of categorical columns:")
for col in df.select_dtypes(exclude="number").columns:
    n_unique = df[col].nunique()
    print(f"  - {col}: {n_unique} unique values")
    if n_unique <= 150:  # short lists (e.g. explicit, track_genre): show them all
        print(f"    {sorted(df[col].dropna().unique().tolist())}")
print()

print("Missing values per column:")
print(df.isna().sum()[lambda s: s > 0].to_string() or "none", "\n")
print("Rows with missing values:")
print(df[df.isna().any(axis=1)].to_string(), "\n")

print(f"Exact duplicate rows (raw, index columns included): {df.duplicated().sum()}")
print(f"Duplicated track_id values (raw): {df['track_id'].duplicated().sum()}\n")

# --------------------------------------------------------------------------
# 3. Drop the two index columns
# --------------------------------------------------------------------------
# Done BEFORE looking for duplicates: the index columns make every row unique,
# so exact duplicates are only visible once they are gone.
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

# --------------------------------------------------------------------------
# 4. Missing values
# --------------------------------------------------------------------------
# Only one row is affected (artists, album_name and track_name are all empty).
# Its audio features and track_id are valid, so the row is kept and the three
# text fields are filled with "Unknown" instead of deleting a usable row.
text_cols = ["artists", "album_name", "track_name"]
df[text_cols] = df[text_cols].fillna("Unknown")

# --------------------------------------------------------------------------
# 5. Exact duplicate rows
# --------------------------------------------------------------------------
n_before = len(df)
df = df.drop_duplicates()
print(f"Exact duplicate rows removed: {n_before - len(df)}  -> {len(df)} rows")

# --------------------------------------------------------------------------
# 6. Duplicate track_id values
# --------------------------------------------------------------------------
# The same track appears once per genre (same track_id, different track_genre),
# and for some tracks the popularity also differs between those rows.
# If they stayed, the same song could land in both the train and the test set
# of a machine-learning split and inflate the model's score (data leakage).
#
# RULE: keep exactly ONE row per track_id - the row with the HIGHEST popularity.
#   Why: popularity is the usual prediction target, and the highest value is the
#   best estimate of how popular the track really is (lower values come from
#   the same track listed under another genre/album release).
#   Ties: the first row in the original file order is kept (stable sort), so the
#   result is deterministic and reproducible.
#   Consequence: track_genre holds only the genre of the kept row; a track's
#   other genres are not represented.
n_before = len(df)
n_pop_conflict = int((df.groupby("track_id")["popularity"].nunique() > 1).sum())
df = (
    df.sort_values("popularity", ascending=False, kind="stable")
      .drop_duplicates(subset="track_id", keep="first")
      .sort_index()  # restore original file order
      .reset_index(drop=True)
)
print(f"track_ids with conflicting popularity: {n_pop_conflict}")
print(f"Rows removed by the one-row-per-track_id rule: {n_before - len(df)}  -> {len(df)} rows\n")

# --------------------------------------------------------------------------
# 7. Checks and save
# --------------------------------------------------------------------------
assert df["track_id"].duplicated().sum() == 0, "track_id still duplicated"
assert df.isna().sum().sum() == 0, "missing values remain"
assert df.duplicated().sum() == 0, "exact duplicates remain"

print("=" * 70)
print("CLEAN DATA")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Duplicated track_id: {df['track_id'].duplicated().sum()}")
print(f"Missing values: {df.isna().sum().sum()}")

df.to_csv(CLEAN_PATH, index=False)
print(f"\nSaved {CLEAN_PATH}")
