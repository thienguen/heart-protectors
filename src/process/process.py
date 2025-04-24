import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv("../../data/heart_disease.csv")
df = pd.DataFrame(data)

print(df.columns)

# Index(['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
#        'Exercise Habits', 'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI',
#        'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
#        'Alcohol Consumption', 'Stress Level', 'Sleep Hours',
#        'Sugar Consumption', 'Triglyceride Level', 'Fasting Blood Sugar',
#        'CRP Level', 'Homocysteine Level', 'Heart Disease Status'],
#       dtype='object')

print("Data Overview:")
print(df.info())
print(df.describe())
print(df.head())
print(
    "----------------------------------------------------------------------------------------------"
)

# Data Overview:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10000 entries, 0 to 9999
# Data columns (total 21 columns):
#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   Age                   9971 non-null   float64
#  1   Gender                9981 non-null   object
#  2   Blood Pressure        9981 non-null   float64
#  3   Cholesterol Level     9970 non-null   float64
#  4   Exercise Habits       9975 non-null   object
#  5   Smoking               9975 non-null   object
#  6   Family Heart Disease  9979 non-null   object
#  7   Diabetes              9970 non-null   object
#  8   BMI                   9978 non-null   float64
#  9   High Blood Pressure   9974 non-null   object
#  10  Low HDL Cholesterol   9975 non-null   object
#  11  High LDL Cholesterol  9974 non-null   object
#  12  Alcohol Consumption   7414 non-null   object
#  13  Stress Level          9978 non-null   object
#  14  Sleep Hours           9975 non-null   float64
#  15  Sugar Consumption     9970 non-null   object
#  16  Triglyceride Level    9974 non-null   float64
#  17  Fasting Blood Sugar   9978 non-null   float64
#  18  CRP Level             9974 non-null   float64
#  19  Homocysteine Level    9980 non-null   float64
#  20  Heart Disease Status  10000 non-null  object
# dtypes: float64(9), object(12)
# memory usage: 1.6+ MB
# None
#                Age  Blood Pressure  Cholesterol Level          BMI  Sleep Hours  Triglyceride Level  Fasting Blood Sugar    CRP Level  Homocysteine Level
# count  9971.000000     9981.000000        9970.000000  9978.000000  9975.000000         9974.000000          9978.000000  9974.000000         9980.000000
# mean     49.296259      149.757740         225.425577    29.077269     6.991329          250.734409           120.142213     7.472201           12.456271
# std      18.193970       17.572969          43.575809     6.307098     1.753195           87.067226            23.584011     4.340248            4.323426
# min      18.000000      120.000000         150.000000    18.002837     4.000605          100.000000            80.000000     0.003647            5.000236
# 25%      34.000000      134.000000         187.000000    23.658075     5.449866          176.000000            99.000000     3.674126            8.723334
# 50%      49.000000      150.000000         226.000000    29.079492     7.003252          250.000000           120.000000     7.472164           12.409395
# 75%      65.000000      165.000000         263.000000    34.520015     8.531577          326.000000           141.000000    11.255592           16.140564
# max      80.000000      180.000000         300.000000    39.996954     9.999952          400.000000           160.000000    14.997087           19.999037
#     Age  Gender  Blood Pressure  Cholesterol Level Exercise Habits Smoking  ... Sugar Consumption Triglyceride Level  Fasting Blood Sugar  CRP Level Homocysteine Level Heart Disease Status
# 0  56.0    Male           153.0              155.0            High     Yes  ...            Medium              342.0                  NaN  12.969246          12.387250                   No
# 1  69.0  Female           146.0              286.0            High      No  ...            Medium              133.0                157.0   9.355389          19.298875                   No
# 2  46.0    Male           126.0              216.0             Low      No  ...               Low              393.0                 92.0  12.709873          11.230926                   No
# 3  32.0  Female           122.0              293.0            High     Yes  ...              High              293.0                 94.0  12.509046           5.961958                   No
# 4  60.0    Male           166.0              242.0             Low     Yes  ...              High              263.0                154.0  10.381259           8.153887                   No

# [5 rows x 21 columns]

# 2. Check for missing values
# missing_values = df.isnull().sum()
# print("\nMissing values in each column:")
# print(missing_values)
# print(
#     "----------------------------------------------------------------------------------------------"
# )

# Missing values in each column:
# Age                       29
# Gender                    19
# Blood Pressure            19
# Cholesterol Level         30
# Exercise Habits           25
# Smoking                   25
# Family Heart Disease      21
# Diabetes                  30
# BMI                       22
# High Blood Pressure       26
# Low HDL Cholesterol       25
# High LDL Cholesterol      26
# Alcohol Consumption     2586
# Stress Level              22
# Sleep Hours               25
# Sugar Consumption         30
# Triglyceride Level        26
# Fasting Blood Sugar       22
# CRP Level                 26
# Homocysteine Level        20
# Heart Disease Status       0
# dtype: int64
# -----------------------------------------------------------

# Drop 'Alcohol Consumption' column because it contain many missing Data
df = df.drop(columns=["Alcohol Consumption"])

# Check for missing values after dropping the column
# print("\nMissing values after dropping 'Alcohol Consumption' column:")
# print(df.isnull().sum())
# print(
#     "----------------------------------------------------------------------------------------------"
# )
# First, it identifies the numerical columns (e.g., Age, Blood Pressure)
# and imputes their missing values with the mean value of each column, using SimpleImputer.
# Then, it identifies the categorical columns (e.g., Gender, Smoking) and
# imputes their missing values with the most frequent value (mode) in each column.
# The SimpleImputer is used for both numerical and categorical features to ensure that no missing values rema> in,
# improving the dataset for further analysis or modeling :


# 1. For numerical columns - impute with mean
numerical_columns = df.select_dtypes(include=[np.number]).columns
# Fill missing values with the mean of each column
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# 2. For categorical columns - impute with mode (most frequent value)
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
# Fill missing values with the most frequent value (mode) for each column
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check if any missing values remain
# print("\nAfter handling missing values:")
# print(df.isnull().sum())

# Convert data types if necessary (e.g., 'Gender' from object to category)
df["Gender"] = df["Gender"].astype("category")
print(df.info())


# First, it separates the data into features (X) and the target variable (y) for classification.
# The features are categorized into numerical and categorical types.
# Categorical variables are also one-hot encoded, which means each category is
# turned into separate binary columns. After transforming the data, the code uses the
# ANOVA F-test (via SelectKBest)
# to evaluate the importance of each feature in predicting the target variable,
# sorting them by their statistical scores.
# Finally, it prints the sorted features, helping identify which variables are most relevant for classification.

# 6. Identify relevant features using statistical tests (e.g., ANOVA F-value for classification)
# Separate the features (X) and target variable (y)
X = df.drop(
    columns=["Heart Disease Status"]
)  # Features (all columns except 'Heart Disease Status')
y = df["Heart Disease Status"]  # Target variable ('Heart Disease Status')

# Identify categorical and numerical features
categorical_features = [
    "Gender",
    "Exercise Habits",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
    "Stress Level",
    "Sleep Hours",
    "Sugar Consumption",
    "Triglyceride Level",
    "Fasting Blood Sugar",
    "CRP Level",
    "Homocysteine Level",
]
numerical_features = ["Age", "Blood Pressure", "Cholesterol Level", "BMI"]


# # Apply the preprocessor to transform the data
# X_processed = preprocessor.fit_transform(X)

# # Get the column names for the categorical features after one-hot encoding
# # OneHotEncoder creates binary columns for each unique category in categorical features
# ohe_columns = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)

# # Combine the original numerical columns and the new one-hot encoded columns
# columns = numerical_features + ohe_columns.tolist()

# # Create a DataFrame with the transformed data
# X_processed_df = pd.DataFrame(X_processed, columns=columns)

# # 7. Select the best features using SelectKBest (ANOVA F-statistic)
# selector = SelectKBest(score_func=f_classif, k='all')  # Select all features
# selector.fit(X_processed_df, y)  # Fit the selector with the transformed data

# # Get the scores for each feature and sort them
# feature_scores = pd.DataFrame({
#     'Feature': columns,  # Feature names (after encoding)
#     'Score': selector.scores_  # ANOVA F-statistic scores
# })

# # Sort the features by score in descending order
# sorted_features = feature_scores.sort_values(by='Score', ascending=False)

# # Print the relevant features based on the ANOVA F-value
# print("\nRelevant Features based on ANOVA F-value:")
# print(sorted_features)

# 8. Data splitting (Training and Validation sets)
# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# You can print the shape of the datasets to confirm
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Training set size: (8000, 19)
# Test set size: (2000, 19)

# Save the cleaned data
df.to_csv('cleaned_heart_disease.csv', index=False)