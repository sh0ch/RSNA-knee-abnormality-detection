"""Competition-wide constants."""

from __future__ import annotations

# Official competition slug on Kaggle
COMPETITION_SLUG = "rsna-knee-abnormality-detection"

# Canonical label names used in code, metrics, and models (snake_case)
TARGET_LABELS: list[str] = [
    "acl_tear",
    "mcl_tear",
    "medial_meniscus_injury",
    "lateral_meniscus_injury",
    "medial_osteoarthritis",
    "lateral_osteoarthritis",
    "patellofemoral_osteoarthritis",
    "joint_effusion",
    "synovitis",
    "bakers_cyst",
    "bone_contusion",
    "fracture",
]

# Label columns as they appear in Kaggle train.csv / sample_submission.csv
KAGGLE_LABEL_COLUMNS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Canonical name -> Kaggle submission column
SUBMISSION_LABEL_COLUMNS: dict[str, str] = dict(
    zip(TARGET_LABELS, KAGGLE_LABEL_COLUMNS, strict=True)
)

# Human-readable names for logging and reports
TARGET_DISPLAY_NAMES: dict[str, str] = {
    "acl_tear": "ACL Tear",
    "mcl_tear": "MCL Tear",
    "medial_meniscus_injury": "Medial Meniscus Injury",
    "lateral_meniscus_injury": "Lateral Meniscus Injury",
    "medial_osteoarthritis": "Medial Osteoarthritis",
    "lateral_osteoarthritis": "Lateral Osteoarthritis",
    "patellofemoral_osteoarthritis": "Patellofemoral Osteoarthritis",
    "joint_effusion": "Joint Effusion",
    "synovitis": "Synovitis",
    "bakers_cyst": "Baker's Cyst",
    "bone_contusion": "Bone Contusion",
    "fracture": "Fracture",
}

# CSV column names (canonical after normalization)
STUDY_ID_COL = "StudyInstanceUID"
SERIES_ID_COL = "SeriesInstanceUID"
REPORT_COL = "Report"
PATIENT_SEX_COL = "PatientSex"
FLUID_COL = "fluid_sensitive"
FAT_SUPPRESSION_COL = "fat_suppression"
ANATOMICAL_PLANE_COL = "anatomical_plane"

# DICOM layout: {series_root}/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm
TRAIN_SERIES_DIR = "train_series"
TEST_SERIES_DIR = "test_series"
TRAIN_CSV = "train.csv"
TRAIN_SERIES_CSV = "train_series.csv"
TEST_CSV = "test.csv"
TEST_SERIES_CSV = "test_series.csv"
SAMPLE_SUBMISSION_CSV = "sample_submission.csv"

# Typical slice count per series (from competition data page)
TYPICAL_SLICES_PER_SERIES = 30
MIN_SLICES_PER_SERIES = 1
MAX_SLICES_PER_SERIES = 512
