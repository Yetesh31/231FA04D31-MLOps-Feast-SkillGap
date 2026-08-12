from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64

sample = Entity(
    name="sample",
    join_keys=["sample_id"],
    description="CSE curriculum industry skill gap sample"
)

cse_source = FileSource(
    name="cse_source",
    path="data/cse_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

cse_feature_view = FeatureView(
    name="cse_skill_features",
    entities=[sample],
    ttl=timedelta(days=50000),
    schema=[
        Field(name="alignment_score", dtype=Float32),
        Field(name="student_score", dtype=Float32),
        Field(name="industry_score", dtype=Float32),
        Field(name="missing_skills", dtype=Int64),
        Field(name="training_hours", dtype=Float32),
        Field(name="gap_level_encoded", dtype=Int64)
    ],
    source=cse_source,
    online=True
)

cse_skill_gap_service = FeatureService(
    name="cse_skill_gap_service",
    features=[cse_feature_view]
)
