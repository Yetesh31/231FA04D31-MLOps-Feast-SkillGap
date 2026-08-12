# Curriculum-Industry Skill Feature Store Using Feast

GitHub Repository: https://github.com/Yetesh31/231FA04D31-MLOps-Feast-SkillGap

## Student Details

**Name:** Yetesh  
**Register Number:** 231FA04D31  
**Section:** 9  

## Problem Statement

The objective of this project is to build a curriculum-industry skill-gap feature store using Feast.

The project uses the curriculum-industry skill-gap dataset created for the previous activity. The dataset contains curriculum skills, student skills, industry-required skills, skill gaps, target roles, curriculum-industry alignment, student skill scores, industry demand scores, missing skill counts, and recommended training hours.

The project demonstrates feature engineering, Feast entity creation, Feast data source creation, FeatureView creation, registration using `feast apply`, historical feature retrieval, materialization, online feature retrieval, and use of Feast features in a simple machine-learning model.

## Dataset

The dataset file is `data/D31 CSE dataset.csv`.

It contains 2,000 rows and 13 columns.

The dataset contains 54 unique skills across the curriculum, student and industry skill fields.

### Columns

| Column | Meaning |
|---|---|
| sample_id | Unique sample identifier |
| degree_domain | Degree/domain |
| curriculum_skills | Skills included in curriculum |
| student_skills | Skills possessed by student |
| industry_required_skills | Skills required by industry |
| skill_gap | Identified skill gap |
| gap_level | Skill-gap severity |
| target_role | Target role |
| curriculum_industry_alignment | Curriculum-industry alignment score |
| student_skill_score | Student skill score |
| industry_demand_score | Industry demand score |
| missing_skill_count | Number of missing skills |
| training_hours_recommended | Recommended training hours |

### Target

The machine-learning target is `gap_level`.

| Gap Level | Encoded Value |
|---|---:|
| Low | 0 |
| Medium | 1 |
| High | 2 |

### How Entries Were Created

The dataset contains records representing curriculum skills, student skills, industry-required skills, skill gaps, target roles, alignment scores, student skill scores, industry demand scores, missing skill counts and recommended training hours.

## Feature Engineering

The raw categorical target is encoded using:

```text
Low = 0
Medium = 1
High = 2
```

The following Feast features are created:

| Feast Feature | Source | Meaning |
|---|---|---|
| alignment_score | curriculum_industry_alignment | Curriculum-industry alignment score |
| student_score | student_skill_score | Student skill score |
| industry_score | industry_demand_score | Industry demand score |
| missing_skills | missing_skill_count | Number of missing skills |
| training_hours | training_hours_recommended | Recommended training hours |
| gap_level_encoded | gap_level | Encoded gap level |

Missing numerical values, if present, are filled using the corresponding column median before feature creation.

## Feast Architecture

```text
Original Dataset
      |
      v
Feature Engineering
      |
      v
Parquet Offline Data
      |
      v
Feast FeatureView
      |
      +---------------------------+
      |                           |
      v                           v
Historical Features         Materialization
      |                           |
      v                           v
Model Training             Online Store
                                  |
                                  v
                           Online Retrieval
                                  |
                                  v
                              Prediction
```

## Implementation

### Entity

The Feast entity is `sample` and its join key is `sample_id`. Each sample represents one curriculum-industry skill-gap record.

### Data Source

The Feast data source is `cse_source`, backed by `data/cse_features.parquet`. The repository also includes `data/cse_features.csv` as an inspection copy; `prepare_features.py` creates the Parquet file when PyArrow is installed.

### FeatureView

The FeatureView is `cse_skill_features` and contains the six features listed in the Feature Engineering section.

### Feature Service

The FeatureService is `cse_skill_gap_service`.

### Historical Retrieval

Historical features are retrieved with:

```python
store.get_historical_features(
    entity_df=labels,
    features=service
).to_df()
```

### Model

A Decision Tree Classifier is used as a baseline model with:

```python
DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
```

The model uses:

```text
alignment_score
student_score
industry_score
missing_skills
training_hours
```

### Online Retrieval

Online features are retrieved using:

```python
store.get_online_features(
    features=service,
    entity_rows=[{"sample_id": 25}]
).to_dict()
```

## Commands

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Feast feature data

```bash
python prepare_features.py
```

### 3. Register Feast definitions

```bash
feast apply
```

### 4. Materialize the features

```bash
feast materialize 2025-01-01T00:00:00 2025-01-02T00:00:00
```

### 5. Retrieve historical features and train the model

```bash
python train.py
```

### 6. Retrieve online features and make a prediction

```bash
python online_prediction.py
```

## Results

### Historical Feature Output

The historical output is saved to:

`results/historical_features.csv`

It contains the sample identifier, timestamps and Feast feature values retrieved for historical training.

### Model Accuracy

Using the five numerical input features, an 80/20 stratified split, `random_state=42`, and a Decision Tree with `max_depth=4`, the test accuracy is **42.75%**.

The classification report is stored in:

`results/model_results.txt`

### Online Feature Output

For `sample_id = 25`, the online feature values are:

```text
alignment_score = 6
student_score = 0
industry_score = 7
missing_skills = 3
training_hours = 12
```

The online output is saved to:

`results/online_features.csv`

### Final Prediction

For `sample_id = 25`, the final Decision Tree prediction is:

```text
Predicted Skill Gap: Medium
```

The result is saved to:

`results/final_prediction.txt`

## Required Analysis

### 1. What is the entity in your Feast implementation?

The entity is `sample`, identified using `sample_id` as the join key.

### 2. List the features stored in your FeatureView.

The features are `alignment_score`, `student_score`, `industry_score`, `missing_skills`, `training_hours`, and `gap_level_encoded`.

### 3. Explain how one feature was calculated.

`gap_level` is converted from categories to numbers: Low = 0, Medium = 1, and High = 2. The result is stored as `gap_level_encoded`.

### 4. What is the difference between your original dataset and the feature dataset?

The original dataset contains raw curriculum, student, industry, categorical and text information. The feature dataset contains the processed features, entity identifier and timestamps required by Feast.

### 5. What is the purpose of the offline store?

The offline store contains historical feature data and supports historical feature retrieval for model training.

### 6. What is the purpose of the online store?

The online store contains materialized feature values for fast online retrieval during prediction.

### 7. What is the purpose of feast apply?

`feast apply` registers and applies the Feast entity, data source, FeatureView and FeatureService definitions.

### 8. What does materialization do?

Materialization copies feature data from the offline source into the online store so that features can be retrieved quickly for online prediction.

### 9. What is the advantage of retrieving features through Feast instead of manually calculating them separately during training and prediction?

Feast provides reusable and centralized feature definitions. Using the same feature definitions for historical training and online prediction reduces training-serving mismatch and keeps feature retrieval consistent.

### 10. State two limitations of your current dataset.

1. The dataset has limited real-world industry evidence such as continuous employer and job-posting data.
2. The dataset does not contain natural historical timestamps showing how industry requirements change over time. The timestamps in this demonstration are generated for Feast processing.

### 11. State two ways your feature store could be improved when more curriculum and industry evidence becomes available.

1. Add real job descriptions, employer requirements, recruitment data, internship requirements and industry surveys.
2. Add real timestamps and time-based features so that changing industry requirements can be tracked historically.

## Limitations

The current model is a baseline. The 42.75% accuracy indicates that the selected numerical features alone do not fully explain the target. Future versions can include richer skill representations, text embeddings, real job-market data and temporal features.

## Repository Structure

```text
Yetesh-MLOps-Feast-SkillGap/
├── README.md
├── requirements.txt
├── feature_store.yaml
├── features.py
├── prepare_features.py
├── train.py
├── online_prediction.py
├── .gitignore
├── data/
│   ├── D31 CSE dataset.csv
│   ├── cse_features.parquet
│   ├── cse_features.csv
│   └── cse_labels.parquet
├── results/
│   ├── historical_features.csv
│   ├── online_features.csv
│   ├── model_results.txt
│   └── final_prediction.txt
└── screenshots/
```

## Conclusion

This project demonstrates a complete local Feast feature-store workflow for a curriculum-industry skill-gap dataset. The workflow converts raw data into reusable features, registers those features with Feast, retrieves historical features for machine-learning training, materializes features into the online store, retrieves online features and produces a skill-gap prediction.
