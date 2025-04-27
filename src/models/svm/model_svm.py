import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def convertCategoricalToNumerical(data: pd.DataFrame, binaryColumns: list, multiValueColumns: list) -> pd.DataFrame:
    """
        Converts categorical columns to numerical values.

        Args:
            data (pd.DataFrame): The dataframe to be modified
            binaryColumns (list): List of binary categorical columns to convert
            multiValueColumns (list): List of multi-value categorical columns to convert
        
        Returns:
            pd.DataFrame: The modified data with categorical columns converted to numerical values
    """
    multiValueMapping = {'Low': 0, 'Medium': 1, 'High': 2}
    for col in binaryColumns:
        if data[col].dtype == 'object':
            categories = data[col].unique()
            newValues = {categories[0]: 0, categories[1]: 1}
            data[col] = data[col].map(newValues)

    for col in multiValueColumns:
        data[col] = data[col].map(multiValueMapping)

    return data

def splitFeaturesAndTarget(data: pd.DataFrame, target: str) -> tuple:
    """
        Splits the data into training and testing sets.

        Args:
            data (pd.DataFrame): The dataframe containing features and target
            target (str): The name of the target column
        
        Returns:
            tuple: Features (X) and target (y) as separate DataFrames
    """

    # Separate the feature data from the target data
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y

def svmPipeline(numericalFeatures, categoricalFeatures):
    """
        Creates a preprocessing and SVM pipeline for classification.

        Args:
            numericalFeatures (list): List of numerical feature names
            categoricalFeatures (list): List of categorical feature names
        
        Returns:
            Pipeline: A scikit-learn pipeline with preprocessing and SVM classifier
    """
    # Preprocessing Pipeline
    numericalTransformer = StandardScaler()
    categoricalTransformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numericalTransformer, numericalFeatures),
            ("cat", categoricalTransformer, categoricalFeatures),
        ]
    )

    # SVM with RBF
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier",
             SVC(
                kernel="rbf",
                # class_weight="balanced",
                probability=True,
                random_state=42
            )
        ),]
    )

    return model

def findCorrelatedPairs(data: pd.DataFrame, features: list, threshold: float = 0.8):
    """
        Finds and prints pairs of features with high correlation which can affect the results of SVM.

        Args:
            data (pd.DataFrame): The dataframe containing the features.
            features (list): List of feature names to check.
            threshold (float): Correlation threshold above which to flag pairs.
    """

    corr_matrix = data[features].corr().abs()
    # Only look at upper triangle (to avoid duplicate pairs)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    print(f"\nHighly correlated feature pairs (|corr| > {threshold}):")
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            corr_value = upper.loc[row, col]
            if pd.notnull(corr_value) and corr_value > threshold:
                print(f"  {row} <--> {col}: {corr_value:.2f}")
                # Suggest dropping one (arbitrarily, the second)
                to_drop.add(col)
    if to_drop:
        print("\nSuggested features to consider dropping (one from each pair):")
        print(sorted(to_drop))
    else:
        print("  No highly correlated pairs found.")

def main() -> int:
    """
        The entry point of the program

        Args:
            None

        Returns
            None
    """

    # Get data into dataframe
    heartData = pd.read_csv("../../process/cleaned_heart_disease.csv")

    binaryColumns = [
        'Gender',
        'Smoking',
        'Family Heart Disease',
        'Diabetes',
        'High Blood Pressure',
        'Low HDL Cholesterol',
        'High LDL Cholesterol',
        'Heart Disease Status'
    ]
    multiValueColumns = ['Exercise Habits', 'Stress Level', 'Sugar Consumption']

    convertedData = convertCategoricalToNumerical(heartData, binaryColumns, multiValueColumns)

    X, y = splitFeaturesAndTarget(convertedData , "Heart Disease Status")

    categoricalFeatures = [
        "Gender",
        "Exercise Habits",
        "Smoking",
        "Family Heart Disease",
        "Diabetes",
        "High Blood Pressure",
        "Low HDL Cholesterol",
        "High LDL Cholesterol",
        "Stress Level"
    ]
    numericalFeatures = [
        "Age",
        "Blood Pressure",
        "Cholesterol Level",
        "BMI",
        "Sleep Hours",
        "Sugar Consumption",
        "Triglyceride Level",
        "Fasting Blood Sugar",
        "CRP Level",
        "Homocysteine Level"
    ]
    allFeatures = numericalFeatures + categoricalFeatures

    findCorrelatedPairs(convertedData, allFeatures)

    # Split the data into training and testing sets (80% train, 20% test)
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = svmPipeline(numericalFeatures, categoricalFeatures)

    model.fit(xTrain, yTrain)

    # Predict the test set
    yPred = model.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)
    confusionMatrix = confusion_matrix(yTest, yPred)
    classificationReport = classification_report(yTest, yPred)

    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(confusionMatrix)
    print("\nClassification Report:")
    print(classificationReport)

    kFoldScored = []

    for k in range(5, 21):
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        kFoldScored.append((k, scores))

    print("\nK-Fold cross-validation results:")
    for k, scores in kFoldScored:
        print(f"{k}-fold: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

    return 0

if __name__ == "__main__":
    main()

