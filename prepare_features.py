import os
import pandas as pd

DATASET = "data/D31 CSE dataset.csv"

os.makedirs("data", exist_ok=True)

_df = pd.read_csv(DATASET)

_df["sample_id"] = _df["sample_id"].astype("int64")

for column in [
    "curriculum_industry_alignment",
    "student_skill_score",
    "industry_demand_score",
    "missing_skill_count",
    "training_hours_recommended"
]:
    _df[column] = _df[column].fillna(_df[column].median())

gap_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

_df["gap_level_encoded"] = (
    _df["gap_level"].map(gap_mapping).fillna(0).astype("int64")
)

_df["alignment_score"] = _df["curriculum_industry_alignment"].astype("float32")
_df["student_score"] = _df["student_skill_score"].astype("float32")
_df["industry_score"] = _df["industry_demand_score"].astype("float32")
_df["missing_skills"] = _df["missing_skill_count"].astype("int64")
_df["training_hours"] = _df["training_hours_recommended"].astype("float32")

base_time = pd.Timestamp("2025-01-01", tz="UTC")
_df["event_timestamp"] = base_time + pd.to_timedelta(_df["sample_id"], unit="s")
_df["created_timestamp"] = _df["event_timestamp"] + pd.Timedelta(seconds=1)

feature_df = _df[
    [
        "sample_id",
        "event_timestamp",
        "created_timestamp",
        "alignment_score",
        "student_score",
        "industry_score",
        "missing_skills",
        "training_hours",
        "gap_level_encoded"
    ]
].copy()

feature_df.to_parquet("data/cse_features.parquet", index=False)

label_df = _df[["sample_id", "event_timestamp", "gap_level_encoded"]].copy()
label_df = label_df.rename(columns={"gap_level_encoded": "gap_level"})
label_df.to_parquet("data/cse_labels.parquet", index=False)

print("Created data/cse_features.parquet")
print("Created data/cse_labels.parquet")
print("Rows:", len(feature_df))
