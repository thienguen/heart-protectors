import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import sys
import os
import missingno as msno

print("Visualizing...\n\n")
# Add the parent directory to the path to import from process.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process.process import df

df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf values with NaN

# Visualize missing values
plt.figure(figsize=(10, 8))
ax = msno.matrix(df)
fig = ax.get_figure()
fig.savefig("../../paper/pictures/missing_values_matrix.png")
plt.close(fig)

plt.figure(figsize=(10, 8))
ax = msno.heatmap(df)
fig = ax.get_figure()
fig.savefig("../../paper/pictures/missing_values_heatmap.png")
plt.close(fig)

# Check for outliers
plt.figure(figsize=(14, 8))
sns.boxplot(data=df.select_dtypes(include=[np.number]))
plt.title("Boxplot of Numerical Features to Detect Outliers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../../paper/pictures/outliers.png")
plt.close(fig)

#  Plot distributions for numerical features
numerical_features = ["Age", "Blood Pressure", "Cholesterol Level", "BMI"]

# Create subplots
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(
        df[feature], kde=True
    )  # Visualizing distribution for each numerical feature
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.savefig("../../paper/pictures/numerical_features_distribution.png")
plt.show()


# Categorize Blood Pressure into categories
bins = [0, 120, 140, 200]
labels = ["Low", "Normal", "High"]
df["Blood Pressure Category"] = pd.cut(
    df["Blood Pressure"], bins=bins, labels=labels, right=False
)

# Count plot to show the distribution of Blood Pressure categories vs Heart Disease Status
# individuals with higher blood pressure tend to show a higher incidence of heart disease (Heart Disease Status = 1).
plt.figure(figsize=(10, 6))
sns.countplot(x="Blood Pressure Category", hue="Heart Disease Status", data=df)
plt.title("Blood Pressure Categories vs Heart Disease Status")
plt.xlabel("Blood Pressure Category")
plt.ylabel("Count")
plt.savefig(
    "../../paper/pictures/blood_pressure_categories_vs_heart_disease_status.png"
)
plt.show()

# Visualization for smoking vs heart disease status
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Smoking", hue="Heart Disease Status", palette="Paired")
plt.title("Smoking vs Heart Disease Status")
plt.xlabel("Smoking")
plt.ylabel("Count")
plt.savefig("../../paper/pictures/smoking_vs_heart_disease_status.png")
plt.show()
