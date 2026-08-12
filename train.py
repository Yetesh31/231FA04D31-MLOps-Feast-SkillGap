import os
import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("results", exist_ok=True)

labels = pd.read_parquet("data/cse_labels.parquet")
store = FeatureStore(repo_path=".")
service = store.get_feature_service("cse_skill_gap_service")

training_data = store.get_historical_features(
    entity_df=labels,
    features=service
).to_df()

training_data.to_csv("results/historical_features.csv", index=False)

features = [
    "alignment_score",
    "student_score",
    "industry_score",
    "missing_skills",
    "training_hours"
]

X = training_data[features]
y = training_data["gap_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

with open("results/model_results.txt", "w") as file:
    file.write("Model: Decision Tree Classifier\n")
    file.write("Test size: 20%\n")
    file.write("Random state: 42\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(report)
