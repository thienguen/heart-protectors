import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold


def knnTestAccuracy(scaleTrainX, trainY, scaleTestX, testY, k):
    # Do the KNN clasification
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(scaleTrainX, trainY)
    prediction = knn.predict(scaleTestX)

    # Convert the result and testing sets to arrays
    predictionList = list(prediction)
    testYList = list(testY)

    # Get the count of correct predictions
    count = 0

    for i in range(len(predictionList)):
        if predictionList[i] == testYList[i]:
            count += 1

    # Get and return the accuracy
    accuracy = count / len(predictionList)

    return accuracy


# Read the data
data = pd.read_csv("../../process/cleaned_heart_disease.csv")

# Preprocess the data so that strings are converted to numbers
binaryColumns = [
    "Gender",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
    "Heart Disease Status",
]

# Handling columns with low, med, and high
multiValueColumns = ["Exercise Habits", "Stress Level", "Sugar Consumption"]
multiValues = {"Low": 0, "Medium": 1, "High": 2}

for col in binaryColumns:
    # Check for strings
    if data[col].dtype == "object":
        categories = data[col].unique()
        newValues = {categories[0]: 0, categories[1]: 1}
        data[col] = data[col].map(newValues)

# Convert columns with multiple values to numbers
for col in multiValueColumns:
    print(f" Mapping for {col}: {multiValues}")
    data[col] = data[col].map(multiValues)

# Show new data
print("\nColumns after encoding:")
print(data.columns)
print("\nFirst few rows after encoding:")
print(data.head())

# Get the daa to test and the labels
x = data.drop(columns=["Heart Disease Status"])
y = data["Heart Disease Status"]


# Split the data. 20% for testing, 80% for training
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=0)

# Normalize the scale of the x data
scaler = StandardScaler()
scaleTrainX = scaler.fit_transform(trainX)
scaleTestX = scaler.transform(testX)

# Print percentages for different k values
for i in range(10):
    print(f"{knnTestAccuracy(scaleTrainX, trainY, scaleTestX, testY, i + 1):.0%}")

average = 0
# Add averages of various K values (1-10)
for i in range(10):
    average += knnTestAccuracy(scaleTrainX, trainY, scaleTestX, testY, i + 1)

print(f"{(average / 10):.0%}")

# Cross-validation using k-folds from 5 to 20

print("\nK-Fold Cross-Validation Accuracies (k=5 to 20):")
for folds in range(5, 21):
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    foldAccuracies = []

    for trainIndex, testIndex in kf.split(x):
        # Split the data into training and testing for this fold
        foldTrainX, foldTestX = x.iloc[trainIndex], x.iloc[testIndex]
        foldTrainY, foldTestY = y.iloc[trainIndex], y.iloc[testIndex]

        # Normalize the scale of the x data for this fold
        scaler = StandardScaler()
        scaleFoldTrainX = scaler.fit_transform(foldTrainX)
        scaleFoldTestX = scaler.transform(foldTestX)

        # Use k=5 for KNN in cross-validation
        acc = knnTestAccuracy(scaleFoldTrainX, foldTrainY, scaleFoldTestX, foldTestY, 5)
        foldAccuracies.append(acc)

    # Get and print the average accuracy for this number of folds
    averageAccuracy = sum(foldAccuracies) / len(foldAccuracies)
    print(f"{folds}-fold: {averageAccuracy:.0%}")
