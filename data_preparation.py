# ============================================================
# CYBERSECURITY ATTACK DATASET
# DATA PREPARATION
# ============================================================

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

FILE_PATH = "cyber_dataset/Attack_Dataset.csv"

TARGET = "Category"

MIN_SAMPLES_PER_CLASS = 100

TEST_SIZE = 0.20

RANDOM_STATE = 42


# Text features selected for the classification problem
TEXT_FEATURES = [
    "Title",
    "Scenario Description",
    "Tools Used",
    "Attack Steps",
    "Target Type",
    "Vulnerability",
    "MITRE Technique",
    "Impact",
    "Detection Method",
    "Tags"
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n====================================================")
print("1. LOADING DATASET")
print("====================================================\n")

try:

    df = pd.read_csv(FILE_PATH)

except FileNotFoundError:

    print(f"ERROR: Dataset not found: {FILE_PATH}")
    raise


print("Dataset loaded successfully.")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 3. REMOVE CSV ARTIFACT COLUMNS
# ============================================================

print("\n====================================================")
print("2. REMOVING CSV ARTIFACTS")
print("====================================================\n")


unnamed_columns = [
    column
    for column in df.columns
    if str(column).startswith("Unnamed:")
]


if unnamed_columns:

    print("Removing:")

    for column in unnamed_columns:
        print(f"- {column}")

    df = df.drop(
        columns=unnamed_columns,
        errors="ignore"
    )

else:

    print("No Unnamed columns detected.")


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 5. REMOVE EMPTY ROWS / COLUMNS
# ============================================================

print("\n====================================================")
print("3. REMOVING EMPTY DATA")
print("====================================================\n")


original_rows = len(df)


df = df.dropna(
    axis=0,
    how="all"
)

df = df.dropna(
    axis=1,
    how="all"
)


print(
    f"Completely empty rows removed: "
    f"{original_rows - len(df)}"
)


# ============================================================
# 6. REMOVE DUPLICATED ROWS
# ============================================================

print("\n====================================================")
print("4. DUPLICATE HANDLING")
print("====================================================\n")


duplicates_before = df.duplicated().sum()

print(
    f"Duplicated rows detected: "
    f"{duplicates_before}"
)


df = df.drop_duplicates().copy()


print(
    f"Rows after duplicate removal: "
    f"{len(df)}"
)


# ============================================================
# 7. CHECK TARGET
# ============================================================

print("\n====================================================")
print("5. TARGET VALIDATION")
print("====================================================\n")


if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


print(f"Target variable: {TARGET}")

print(
    f"Original number of classes: "
    f"{df[TARGET].nunique()}"
)


# ============================================================
# 8. REMOVE MISSING TARGET VALUES
# ============================================================

missing_target = df[TARGET].isna().sum()


print(
    f"Missing target values: "
    f"{missing_target}"
)


df = df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# 9. CLEAN TARGET STRINGS
# ============================================================

# Remove spaces at the beginning/end of category names.

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
)


# Remove empty target strings if they exist.

df = df[
    df[TARGET] != ""
].copy()


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

print("\n====================================================")
print("6. ORIGINAL CLASS DISTRIBUTION")
print("====================================================\n")


class_counts = (
    df[TARGET]
    .value_counts()
)


print(class_counts.to_string())


# ============================================================
# 11. HANDLE RARE CLASSES
# ============================================================

print("\n====================================================")
print("7. RARE CLASS HANDLING")
print("====================================================\n")


valid_classes = class_counts[
    class_counts >= MIN_SAMPLES_PER_CLASS
].index


rare_classes = class_counts[
    class_counts < MIN_SAMPLES_PER_CLASS
]


print(
    f"Minimum samples required per class: "
    f"{MIN_SAMPLES_PER_CLASS}"
)

print(
    f"Classes before filtering: "
    f"{len(class_counts)}"
)

print(
    f"Classes retained: "
    f"{len(valid_classes)}"
)

print(
    f"Rare classes excluded from first experiment: "
    f"{len(rare_classes)}"
)


rows_before_filtering = len(df)


df = df[
    df[TARGET].isin(valid_classes)
].copy()


print(
    f"\nRows before class filtering: "
    f"{rows_before_filtering}"
)

print(
    f"Rows after class filtering: "
    f"{len(df)}"
)

print(
    f"Rows excluded: "
    f"{rows_before_filtering - len(df)}"
)


# ============================================================
# 12. CHECK AVAILABLE TEXT FEATURES
# ============================================================

print("\n====================================================")
print("8. FEATURE SELECTION")
print("====================================================\n")


available_features = [
    column
    for column in TEXT_FEATURES
    if column in df.columns
]


missing_features = [
    column
    for column in TEXT_FEATURES
    if column not in df.columns
]


print("Selected text features:\n")

for feature in available_features:
    print(f"- {feature}")


if missing_features:

    print("\nRequested features not found:")

    for feature in missing_features:
        print(f"- {feature}")


if not available_features:

    raise ValueError(
        "No selected text features were found "
        "in the dataset."
    )


# ============================================================
# 13. MISSING VALUES IN FEATURES
# ============================================================

print("\n====================================================")
print("9. MISSING FEATURE VALUES")
print("====================================================\n")


for feature in available_features:

    missing = df[feature].isna().sum()

    percentage = (
        df[feature].isna().mean()
        * 100
    )

    print(
        f"{feature}: "
        f"{missing} missing "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 14. HANDLE MISSING TEXT
# ============================================================

# Missing textual information is replaced with an empty string.
#
# This allows us to combine text fields without deleting an
# entire record simply because one text field is missing.

df[available_features] = (
    df[available_features]
    .fillna("")
)


# ============================================================
# 15. NORMALIZE TEXT FIELDS
# ============================================================

print("\n====================================================")
print("10. BASIC TEXT CLEANING")
print("====================================================\n")


for feature in available_features:

    df[feature] = (
        df[feature]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )


print("Basic text cleaning completed.")


# ============================================================
# 16. FEATURE ENGINEERING
# ============================================================

# The dataset contains several textual columns.
#
# Instead of treating every column independently for the
# first baseline model, they are combined into one document.
#
# Example:
#
# Title + Scenario Description + Tools Used + ...
#
# This combined document will later be converted into
# numerical TF-IDF features.


print("\n====================================================")
print("11. FEATURE ENGINEERING")
print("====================================================\n")


df["Combined_Text"] = (
    df[available_features]
    .agg(
        " ".join,
        axis=1
    )
    .str.strip()
)


print(
    "Text features combined into: "
    "Combined_Text"
)


# ============================================================
# 17. REMOVE RECORDS WITH NO USABLE TEXT
# ============================================================

empty_text_rows = (
    df["Combined_Text"]
    .str.strip()
    .eq("")
)


number_empty = empty_text_rows.sum()


print(
    f"Rows without usable text: "
    f"{number_empty}"
)


df = df[
    ~empty_text_rows
].copy()


# ============================================================
# 18. CREATE ADDITIONAL TEXT FEATURES
# ============================================================

# These features are useful for understanding the data
# and may be tested later in more advanced experiments.


df["Text_Length"] = (
    df["Combined_Text"]
    .str.len()
)


df["Word_Count"] = (
    df["Combined_Text"]
    .str.split()
    .str.len()
)


print("\nText length statistics:\n")

print(
    df[
        [
            "Text_Length",
            "Word_Count"
        ]
    ]
    .describe()
)


# ============================================================
# 19. DEFINE X AND y
# ============================================================

print("\n====================================================")
print("12. DEFINING FEATURES AND TARGET")
print("====================================================\n")


X = df["Combined_Text"]

y = df[TARGET]


print(
    f"Number of samples: "
    f"{len(X)}"
)

print(
    f"Number of target classes: "
    f"{y.nunique()}"
)


# ============================================================
# 20. TRAIN / TEST SPLIT
# ============================================================

print("\n====================================================")
print("13. TRAIN / TEST SPLIT")
print("====================================================\n")


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)

print(
    f"Training percentage: "
    f"{(1 - TEST_SIZE) * 100:.0f}%"
)

print(
    f"Testing percentage: "
    f"{TEST_SIZE * 100:.0f}%"
)


# ============================================================
# 21. VERIFY STRATIFICATION
# ============================================================

print("\n====================================================")
print("14. STRATIFICATION CHECK")
print("====================================================\n")


train_distribution = (
    y_train
    .value_counts(
        normalize=True
    )
    .mul(100)
)


test_distribution = (
    y_test
    .value_counts(
        normalize=True
    )
    .mul(100)
)


distribution_check = pd.DataFrame({
    "Train (%)":
        train_distribution,

    "Test (%)":
        test_distribution
})


print(
    distribution_check
    .round(2)
    .to_string()
)


# ============================================================
# 22. FEATURE EXTRACTION - TF-IDF
# ============================================================

print("\n====================================================")
print("15. TF-IDF FEATURE EXTRACTION")
print("====================================================\n")


# TF-IDF converts text into numerical features.
#
# Important:
# fit_transform() is performed ONLY on training data.
#
# The test data uses transform() only.
#
# This prevents information from the test set from leaking
# into the training process.


tfidf = TfidfVectorizer(

    # Ignore words appearing in only one document
    min_df=2,

    # Ignore terms appearing in more than 95% of documents
    max_df=0.95,

    # Use individual words and pairs of words
    ngram_range=(1, 2),

    # Limit feature space for the baseline experiment
    max_features=20000,

    # Normalize text
    lowercase=True,

    # Standard English stopwords
    stop_words="english"
)


X_train_tfidf = tfidf.fit_transform(
    X_train
)


X_test_tfidf = tfidf.transform(
    X_test
)


print(
    "TF-IDF transformation completed."
)


print(
    f"\nTraining matrix shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Testing matrix shape: "
    f"{X_test_tfidf.shape}"
)

print(
    f"Vocabulary size: "
    f"{len(tfidf.vocabulary_)}"
)


# ============================================================
# 23. FEATURE SCALING
# ============================================================

print("\n====================================================")
print("16. FEATURE SCALING")
print("====================================================\n")


print(
    "StandardScaler was not applied."
)

print(
    "TF-IDF already produces normalized numerical "
    "representations suitable for the initial "
    "text-classification models."
)


# ============================================================
# 24. IMBALANCED DATA CHECK
# ============================================================

print("\n====================================================")
print("17. CLASS BALANCE AFTER PREPARATION")
print("====================================================\n")


final_class_counts = (
    y.value_counts()
)


print(
    final_class_counts.to_string()
)


largest_class = (
    final_class_counts.max()
)


smallest_class = (
    final_class_counts.min()
)


imbalance_ratio = (
    largest_class
    / smallest_class
)


print(
    f"\nLargest class: "
    f"{largest_class}"
)

print(
    f"Smallest class: "
    f"{smallest_class}"
)

print(
    f"Imbalance ratio: "
    f"{imbalance_ratio:.2f}:1"
)


# ============================================================
# 25. DATA LEAKAGE CHECK
# ============================================================

print("\n====================================================")
print("18. DATA LEAKAGE PREVENTION")
print("====================================================\n")


print(
    "ID is not used as a predictive feature."
)

print(
    "Unnamed CSV artifact columns are removed."
)

print(
    "The target variable is not included "
    "inside Combined_Text."
)

print(
    "TF-IDF is fitted only on training data."
)

print(
    "Test data is transformed using the vocabulary "
    "learned from training data."
)


# ============================================================
# 26. FINAL DATA PREPARATION SUMMARY
# ============================================================

print("\n====================================================")
print("FINAL DATA PREPARATION SUMMARY")
print("====================================================\n")


print(
    f"Target variable: "
    f"{TARGET}"
)

print(
    f"Minimum samples per class: "
    f"{MIN_SAMPLES_PER_CLASS}"
)

print(
    f"Final records: "
    f"{len(df)}"
)

print(
    f"Final classes: "
    f"{y.nunique()}"
)

print(
    f"Text features used: "
    f"{len(available_features)}"
)

print(
    f"Training records: "
    f"{len(X_train)}"
)

print(
    f"Testing records: "
    f"{len(X_test)}"
)

print(
    f"TF-IDF features: "
    f"{X_train_tfidf.shape[1]}"
)

print(
    f"Training matrix: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Testing matrix: "
    f"{X_test_tfidf.shape}"
)

print(
    f"Class imbalance ratio: "
    f"{imbalance_ratio:.2f}:1"
)


print("\n====================================================")
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("READY FOR SUPERVISED LEARNING")
print("====================================================\n")