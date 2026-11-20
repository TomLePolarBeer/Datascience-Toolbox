
"""
Processing_1.py - calculate Spotify indicators.

Input : data/spotify_clean.csv
Output: data/genre_indicators.csv
        data/audio_indicators.csv
        data/feature_correlations.csv
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def process_data():
    print("Starting data processing...")

    # ----------------------------------------------------------------------
    # 1. Load cleaned data
    # ----------------------------------------------------------------------
    tracks = pd.read_csv(DATA_DIR / "spotify_clean.csv")

    print("Cleaned tracks:", len(tracks))

    # Check required columns
    required_columns = [
        "track_id", "track_genre", "popularity",
        "energy", "danceability", "duration_ms",
        "valence", "acousticness", "instrumentalness"
    ]

    missing_columns = [
        col for col in required_columns if col not in tracks.columns
    ]

    assert not missing_columns, f"Missing columns: {missing_columns}"
    assert tracks["track_id"].duplicated().sum() == 0, \
        "Duplicated track IDs found"

    # ----------------------------------------------------------------------
    # 2. Question 2: Which genres perform best,
    #    and what distinguishes their songs?
    # ----------------------------------------------------------------------

    # Calculate genre indicators
    genre_indicators = tracks.groupby(
        "track_genre", as_index=False
    ).agg(
        Average_Popularity=("popularity", "mean"),
        Median_Popularity=("popularity", "median"),
        Popularity_Std=("popularity", "std"),
        Number_of_Tracks=("track_id", "count"),
        Average_Energy=("energy", "mean"),
        Average_Danceability=("danceability", "mean"),
        Average_Valence=("valence", "mean")
    )

    # Sort genres by average popularity
    genre_indicators = genre_indicators.sort_values(
        "Average_Popularity",
        ascending=False
    )

    # Save genre indicators
    genre_indicators.to_csv(
        DATA_DIR / "genre_indicators.csv",
        index=False
    )

    # ----------------------------------------------------------------------
    # 3. Question 1: Which musical characteristics
    #    are associated with higher popularity?
    # ----------------------------------------------------------------------

    # Convert duration from milliseconds to minutes
    tracks["Duration_Minutes"] = tracks["duration_ms"] / 60000

    # Remove zero-duration tracks from duration calculations
    valid_duration = tracks[tracks["duration_ms"] > 0].copy()

    # Group energy and danceability into descriptive categories
    tracks["Energy_Level"] = pd.cut(
        tracks["energy"],
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"]
    )

    tracks["Danceability_Level"] = pd.cut(
        tracks["danceability"],
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"]
    )

    # Average popularity by energy level
    energy_indicators = tracks.groupby(
        "Energy_Level", observed=True, as_index=False
    ).agg(
        Average_Popularity=("popularity", "mean"),
        Number_of_Tracks=("track_id", "count")
    )

    energy_indicators["Feature"] = "Energy"
    energy_indicators = energy_indicators.rename(
        columns={"Energy_Level": "Level"}
    )

    # Average popularity by danceability level
    danceability_indicators = tracks.groupby(
        "Danceability_Level", observed=True, as_index=False
    ).agg(
        Average_Popularity=("popularity", "mean"),
        Number_of_Tracks=("track_id", "count")
    )

    danceability_indicators["Feature"] = "Danceability"
    danceability_indicators = danceability_indicators.rename(
        columns={"Danceability_Level": "Level"}
    )

    # Combine audio indicators
    audio_indicators = pd.concat(
        [energy_indicators, danceability_indicators],
        ignore_index=True
    )

    audio_indicators = audio_indicators[
        ["Feature", "Level", "Average_Popularity", "Number_of_Tracks"]
    ]

    audio_indicators.to_csv(
        DATA_DIR / "audio_indicators.csv",
        index=False
    )

    # ----------------------------------------------------------------------
    # 4. Correlation analysis
    # ----------------------------------------------------------------------

    audio_features = [
        "danceability", "energy", "valence",
        "acousticness", "instrumentalness",
        "speechiness", "liveness", "tempo",
        "Duration_Minutes"
    ]

    # Correlation between each audio feature and popularity
    feature_correlations = valid_duration[
        ["popularity"] + audio_features
    ].corr(numeric_only=True)[["popularity"]]

    feature_correlations = (
        feature_correlations
        .drop(index="popularity")
        .reset_index()
        .rename(columns={
            "index": "Feature",
            "popularity": "Correlation_with_Popularity"
        })
        .sort_values(
            "Correlation_with_Popularity",
            ascending=False
        )
    )

    feature_correlations.to_csv(
        DATA_DIR / "feature_correlations.csv",
        index=False
    )

    # ----------------------------------------------------------------------
    # 5. Display results
    # ----------------------------------------------------------------------

    print("\nData processing completed.")
    print("Genres analyzed:", len(genre_indicators))

    print("\nGenre indicators:")
    print(genre_indicators.head())

    print("\nAudio indicators:")
    print(audio_indicators)

    print("\nFeature correlations:")
    print(feature_correlations)

    print("\nSaved:")
    print("data/genre_indicators.csv")
    print("data/audio_indicators.csv")
    print("data/feature_correlations.csv")


if __name__ == "__main__":
    process_data()
