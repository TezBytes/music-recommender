
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
