# ============================================================
# CYBERSECURITY ATTACK DATASET
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = "cyber_dataset/Attack_Dataset.csv"

try:
    df = pd.read_csv(FILE_PATH)

    print("\n====================================================")
    print("DATASET LOADED SUCCESSFULLY")
    print("====================================================")

except FileNotFoundError:
    print(f"\nERROR: Dataset not found at {FILE_PATH}")
    raise


# ============================================================
# 2. ORIGINAL DATASET INFORMATION
# ============================================================

print("\n====================================================")
print("ORIGINAL DATASET DIMENSIONS")
print("====================================================")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


print("\n====================================================")
print("COLUMN NAMES")
print("====================================================")

for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")


print("\n====================================================")
print("FIRST 5 ROWS")
print("====================================================")

print(df.head())


print("\n====================================================")
print("DATA TYPES")
print("====================================================")

print(df.dtypes)


# ============================================================
# 3. REMOVE CSV ARTIFACT COLUMNS
# ============================================================

print("\n====================================================")
print("CHECKING CSV ARTIFACT COLUMNS")
print("====================================================")

unnamed_columns = [
    column
    for column in df.columns
    if str(column).startswith("Unnamed:")
]

if unnamed_columns:

    print("\nUnnamed columns detected:")

    for column in unnamed_columns:

        missing_percentage = (
            df[column].isna().mean() * 100
        )

        non_null = df[column].notna().sum()

        print(
            f"- {column}: "
            f"{non_null} non-null values | "
            f"{missing_percentage:.2f}% missing"
        )

    df = df.drop(
        columns=unnamed_columns,
        errors="ignore"
    )

    print(
        f"\nRemoved {len(unnamed_columns)} "
        "Unnamed column(s)."
    )

else:

    print("No Unnamed columns detected.")


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 5. REMOVE COMPLETELY EMPTY ROWS AND COLUMNS
# ============================================================

df = df.dropna(
    axis=0,
    how="all"
)

df = df.dropna(
    axis=1,
    how="all"
)


# ============================================================
# 6. DATASET AFTER BASIC CLEANING
# ============================================================

print("\n====================================================")
print("DATASET AFTER BASIC CLEANING")
print("====================================================")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n====================================================")
print("MISSING VALUES")
print("====================================================")

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
)

missing_table = pd.DataFrame({
    "Missing Values": missing_count,
    "Percentage (%)": missing_percentage.round(2)
})

missing_table = missing_table[
    missing_table["Missing Values"] > 0
].sort_values(
    by="Percentage (%)",
    ascending=False
)

if missing_table.empty:

    print("No missing values detected.")

else:

    print(missing_table)


print(
    f"\nTotal missing values: "
    f"{df.isnull().sum().sum()}"
)


# ============================================================
# 8. DUPLICATES
# ============================================================

print("\n====================================================")
print("DUPLICATED ROWS")
print("====================================================")

duplicates = df.duplicated().sum()

duplicate_percentage = (
    duplicates / len(df)
) * 100

print(f"Duplicated rows: {duplicates}")

print(
    f"Duplicate percentage: "
    f"{duplicate_percentage:.2f}%"
)


# ============================================================
# 9. UNIQUE VALUES
# ============================================================

print("\n====================================================")
print("UNIQUE VALUES PER COLUMN")
print("====================================================")

unique_table = pd.DataFrame({
    "Column": df.columns,
    "Unique Values": [
        df[column].nunique(dropna=False)
        for column in df.columns
    ]
})

unique_table = unique_table.sort_values(
    by="Unique Values"
)

print(
    unique_table.to_string(index=False)
)


# ============================================================
# 10. IDENTIFIER COLUMN
# ============================================================

print("\n====================================================")
print("IDENTIFIER ANALYSIS")
print("====================================================")

if "ID" in df.columns:

    print(
        f"ID contains "
        f"{df['ID'].nunique()} unique values."
    )

    print(
        "ID will not be used as a predictive feature "
        "because it is only a record identifier."
    )

else:

    print("No ID column found.")


# ============================================================
# 11. CATEGORY ANALYSIS
# ============================================================

if "Category" not in df.columns:
    raise ValueError(
        "The expected 'Category' column was not found."
    )


category_counts = (
    df["Category"]
    .dropna()
    .value_counts()
)


print("\n====================================================")
print("CATEGORY ANALYSIS")
print("====================================================")

print(
    f"Number of unique categories: "
    f"{df['Category'].nunique()}"
)

print("\nCategory distribution:\n")

print(
    category_counts.to_string()
)


# ============================================================
# 12. TOP 20 CYBERSECURITY CATEGORIES
# ============================================================

print("\n====================================================")
print("GRAPH 1 - TOP 20 CYBERSECURITY CATEGORIES")
print("====================================================")

top_categories = category_counts.head(20)

plt.figure(
    figsize=(12, 8)
)

sns.barplot(
    x=top_categories.values,
    y=top_categories.index
)

plt.title(
    "Top 20 Cybersecurity Categories"
)

plt.xlabel(
    "Number of Samples"
)

plt.ylabel(
    "Category"
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. NUMBER OF SAMPLES PER CATEGORY
# ============================================================

print("\n====================================================")
print("GRAPH 2 - SAMPLES PER CATEGORY")
print("====================================================")

plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    category_counts.values,
    bins=20,
    kde=True
)

plt.title(
    "Distribution of Samples per Cybersecurity Category"
)

plt.xlabel(
    "Number of Samples in Category"
)

plt.ylabel(
    "Number of Categories"
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. CATEGORY CLASS IMBALANCE
# ============================================================

print("\n====================================================")
print("CATEGORY CLASS BALANCE")
print("====================================================")

print(
    f"Largest category: "
    f"{category_counts.index[0]}"
)

print(
    f"Largest category samples: "
    f"{category_counts.iloc[0]}"
)

print(
    f"Smallest category: "
    f"{category_counts.index[-1]}"
)

print(
    f"Smallest category samples: "
    f"{category_counts.iloc[-1]}"
)


print(
    "\nCategories with fewer than 100 samples:"
)

rare_categories = category_counts[
    category_counts < 100
]

print(
    rare_categories.to_string()
)

print(
    f"\nNumber of categories with "
    f"fewer than 100 samples: "
    f"{len(rare_categories)}"
)


# ============================================================
# 15. CATEGORY SIZE GROUPS
# ============================================================

large_categories = (
    category_counts >= 100
).sum()

medium_categories = (
    (category_counts >= 20)
    & (category_counts < 100)
).sum()

small_categories = (
    category_counts < 20
).sum()


print("\n====================================================")
print("CATEGORY SIZE GROUPS")
print("====================================================")

print(
    f"Categories with >= 100 samples: "
    f"{large_categories}"
)

print(
    f"Categories with 20-99 samples: "
    f"{medium_categories}"
)

print(
    f"Categories with < 20 samples: "
    f"{small_categories}"
)


# ============================================================
# 16. CATEGORY SIZE GROUP GRAPH
# ============================================================

category_size_data = pd.DataFrame({
    "Group": [
        "100+ samples",
        "20-99 samples",
        "<20 samples"
    ],
    "Number of Categories": [
        large_categories,
        medium_categories,
        small_categories
    ]
})


print("\n====================================================")
print("GRAPH 3 - CATEGORY SIZE GROUPS")
print("====================================================")

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=category_size_data,
    x="Group",
    y="Number of Categories"
)

plt.title(
    "Cybersecurity Categories by Dataset Size"
)

plt.xlabel(
    "Category Size"
)

plt.ylabel(
    "Number of Categories"
)

plt.tight_layout()

plt.show()


# ============================================================
# 17. TOP TARGET TYPES
# ============================================================

if "Target Type" in df.columns:

    target_counts = (
        df["Target Type"]
        .dropna()
        .value_counts()
        .head(15)
    )

    print("\n====================================================")
    print("TOP 15 TARGET TYPES")
    print("====================================================")

    print(
        target_counts.to_string()
    )


    plt.figure(
        figsize=(12, 7)
    )

    sns.barplot(
        x=target_counts.values,
        y=target_counts.index
    )

    plt.title(
        "Top 15 Cybersecurity Target Types"
    )

    plt.xlabel(
        "Number of Samples"
    )

    plt.ylabel(
        "Target Type"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 18. TOP MITRE ATT&CK TECHNIQUES
# ============================================================

if "MITRE Technique" in df.columns:

    mitre_counts = (
        df["MITRE Technique"]
        .dropna()
        .value_counts()
        .head(15)
    )


    print("\n====================================================")
    print("TOP 15 MITRE ATT&CK TECHNIQUES")
    print("====================================================")

    print(
        mitre_counts.to_string()
    )


    plt.figure(
        figsize=(12, 7)
    )

    sns.barplot(
        x=mitre_counts.values,
        y=mitre_counts.index
    )

    plt.title(
        "Top 15 MITRE ATT&CK Techniques"
    )

    plt.xlabel(
        "Number of Samples"
    )

    plt.ylabel(
        "MITRE Technique"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 19. TOP SOURCES
# ============================================================

if "Source" in df.columns:

    source_counts = (
        df["Source"]
        .dropna()
        .value_counts()
        .head(15)
    )


    print("\n====================================================")
    print("TOP 15 DATA SOURCES")
    print("====================================================")

    print(
        source_counts.to_string()
    )


    plt.figure(
        figsize=(12, 7)
    )

    sns.barplot(
        x=source_counts.values,
        y=source_counts.index
    )

    plt.title(
        "Top 15 Dataset Sources"
    )

    plt.xlabel(
        "Number of Samples"
    )

    plt.ylabel(
        "Source"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 20. TEXT LENGTH ANALYSIS
# ============================================================

print("\n====================================================")
print("TEXT LENGTH ANALYSIS")
print("====================================================")


if "Scenario Description" in df.columns:

    df["Scenario Description Length"] = (
        df["Scenario Description"]
        .fillna("")
        .astype(str)
        .str.len()
    )


    print(
        df["Scenario Description Length"]
        .describe()
    )


    plt.figure(
        figsize=(10, 6)
    )

    sns.histplot(
        df["Scenario Description Length"],
        bins=40,
        kde=True
    )

    plt.title(
        "Distribution of Scenario Description Length"
    )

    plt.xlabel(
        "Number of Characters"
    )

    plt.ylabel(
        "Number of Records"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 21. TITLE LENGTH ANALYSIS
# ============================================================

if "Title" in df.columns:

    df["Title Length"] = (
        df["Title"]
        .fillna("")
        .astype(str)
        .str.len()
    )


    plt.figure(
        figsize=(10, 6)
    )

    sns.histplot(
        df["Title Length"],
        bins=40,
        kde=True
    )

    plt.title(
        "Distribution of Cybersecurity Title Length"
    )

    plt.xlabel(
        "Number of Characters"
    )

    plt.ylabel(
        "Number of Records"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 22. MODEL DATASET SIMULATION
# ============================================================

# For the first classification experiment,
# categories with at least 100 samples are considered.
#
# IMPORTANT:
# This does NOT permanently remove the other categories.
# It creates a separate dataframe for modelling.

MIN_SAMPLES = 100

valid_categories = category_counts[
    category_counts >= MIN_SAMPLES
].index


df_model = df[
    df["Category"].isin(
        valid_categories
    )
].copy()


print("\n====================================================")
print("PROPOSED CLASSIFICATION DATASET")
print("====================================================")

print(
    f"Minimum samples per category: "
    f"{MIN_SAMPLES}"
)

print(
    f"Original number of records: "
    f"{len(df)}"
)

print(
    f"Records available after filtering "
    f"rare categories: "
    f"{len(df_model)}"
)

print(
    f"Original number of categories: "
    f"{df['Category'].nunique()}"
)

print(
    f"Categories available for modelling: "
    f"{df_model['Category'].nunique()}"
)


# ============================================================
# 23. MODEL CATEGORY DISTRIBUTION
# ============================================================

model_category_counts = (
    df_model["Category"]
    .value_counts()
)


plt.figure(
    figsize=(12, 9)
)

sns.barplot(
    x=model_category_counts.values,
    y=model_category_counts.index
)

plt.title(
    "Category Distribution After Removing Rare Classes"
)

plt.xlabel(
    "Number of Samples"
)

plt.ylabel(
    "Cybersecurity Category"
)

plt.tight_layout()

plt.show()


# ============================================================
# 24. POTENTIAL TEXT FEATURES
# ============================================================

potential_text_features = [
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


available_text_features = [
    column
    for column in potential_text_features
    if column in df.columns
]


print("\n====================================================")
print("POTENTIAL MACHINE LEARNING FEATURES")
print("====================================================")

for column in available_text_features:

    print(f"- {column}")


# ============================================================
# 25. TARGET VARIABLE
# ============================================================

print("\n====================================================")
print("PROPOSED TARGET VARIABLE")
print("====================================================")

print("Target: Category")

print(
    "\nThe machine learning task will investigate "
    "whether cybersecurity scenarios can be "
    "classified into their security category "
    "using textual characteristics."
)


# ============================================================
# 26. FINAL EDA SUMMARY
# ============================================================

print("\n====================================================")
print("FINAL EDA SUMMARY")
print("====================================================")

print(
    f"Dataset records: "
    f"{len(df)}"
)

print(
    f"Dataset columns: "
    f"{len(df.columns)}"
)

print(
    f"Cybersecurity categories: "
    f"{df['Category'].nunique()}"
)

print(
    f"Categories with >= 100 samples: "
    f"{large_categories}"
)

print(
    f"Categories with < 100 samples: "
    f"{len(rare_categories)}"
)

print(
    f"Records proposed for first ML experiment: "
    f"{len(df_model)}"
)

print(
    f"Classes proposed for first ML experiment: "
    f"{df_model['Category'].nunique()}"
)

print(
    f"Duplicated rows: "
    f"{duplicates}"
)

print(
    f"Missing values: "
    f"{df.isnull().sum().sum()}"
)


print("\n====================================================")
print("EDA COMPLETED SUCCESSFULLY")
print("====================================================\n")