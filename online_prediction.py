import os
import pandas as pd
from feast import FeatureStore
from sklearn.tree import DecisionTreeClassifier

os.makedirs("results", exist_ok=True)

store = FeatureStore(repo_path=".")
service = store.get_feature_service("cse_skill_gap_service")

sample_id = 25

online_features = store.get_online_features(
    features=service,
    entity_rows=[{"sample_id": sample_id}]
).to_dict()

online_df = pd.DataFrame(online_features)
online_df.to_csv("results/online_features.csv", index=False)

raw = pd.read_csv("data/D31 CSE dataset.csv")
gap_mapping = {"Low": 0, "Medium": 1, "High": 2}

X = raw[[
    "curriculum_industry_alignment",
    "student_skill_score",
    "industry_demand_score",
    "missing_skill_count",
    "training_hours_recommended"
]]
y = raw["gap_level"].map(gap_mapping)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

feature_columns = [
    "alignment_score",
    "student_score",
    "industry_score",
    "missing_skills",
    "training_hours"
]

prediction = model.predict(online_df[feature_columns])[0]
labels = {0: "Low", 1: "Medium", 2: "High"}

with open("results/final_prediction.txt", "w") as file:
    file.write(f"Sample ID: {sample_id}\n")
    file.write(f"Predicted Skill Gap: {labels[int(prediction)]}\n")

print(online_df)
print(f"Predicted Skill Gap: {labels[int(prediction)]}")
