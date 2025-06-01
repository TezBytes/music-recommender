
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def drop_duplicates(df, subset_col="track_id"):
    return df.drop_duplicates(subset=subset_col)

def clean_missing_values(df, required_cols):
    return df.dropna(subset=required_cols)

def convert_types(df):
    df["explicit"] = df["explicit"].astype(bool)
    df["duration_sec"] = df["duration_ms"] / 1000
    return df.drop(columns=["duration_ms"])

def normalize_features(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

GENRE_MAP = {
    "acoustic": "pop",
    "alt-rock":  "rock",
    "tango": "world",
    "ambient": "electronic",
    "afrobeat": "world",
    "cantopop": "pop",
    "bluegrass": "folk",
    "forro": "world",
    "study": "ambient",
    "chicago-house": "electronic",
    "disney": "pop",
    "sleep": "ambient",
    "heavy-metal": "rock",
    "breakbeat": "electronic",
    "black-metal": "rock",
    "j-idol": "pop",
    "happy": "mood",
    "anime": "pop",
    "club": "electronic",
    "comedy": "spoken"
}

def normalize_genre(genre):
    if pd.isna(genre):
        return "unknown"
    return genre.strip().lower()

def map_genre(genre):
    normalized = normalize_genre(genre)
    return GENRE_MAP.get(normalized, "other")

def enrich_with_genre(df, source_col="track_genre", target_col="genre_label"):
    df[target_col] = df[source_col].apply(map_genre)
    return df