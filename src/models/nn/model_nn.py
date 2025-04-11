import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)


def main():
    df = pd.read_csv("./heart.csv", index_col=False, header=0)
    # remove all 0 values from serum cholesterol
    df = df[df["Cholesterol"] != 0]
    y = df["HeartDisease"].reset_index(drop=True)
    X = df.drop(columns="HeartDisease").reset_index(drop=True)

    features_to_encode = [
        5,  # "Sex",
        6,  # "ChestPainType",
        8,  # "RestingECG",
        9,  # "ExerciseAngina",
        10,  # "ST_Slope",
    ]
    features_to_standardize = [
        0,  # Age
        3,  # RestingBP
        4,  # Cholesterol
        7,  # MaxHR
        9,  # Oldpeak
    ]

    categorical_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False), features_to_encode),
        ],
        remainder="passthrough",
        sparse_threshold=0,
    )
    standardize_transformer = ColumnTransformer(
        transformers=[
            ("std", StandardScaler(), features_to_standardize),
        ],
        remainder="passthrough",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Z-norm standardize the training and testing data seperately to avoid weak cheating
    X_train = standardize_transformer.fit_transform(X_train)
    X_test = standardize_transformer.fit_transform(X_test)
    # one hot encoding categorical values
    X_train = categorical_transformer.fit_transform(X_train)
    X_test = categorical_transformer.fit_transform(X_test)
    input_dim = len(X_train[0])
    hidden_layer_nodes = 512
    output_layer_nodes = 2
    hidden_layer_activation = "relu"
    output_layer_activation = "softmax"
    y_train_matrix = to_categorical(y_train, 2)
    y_test_matrix = to_categorical(y_test, 2)

    model = Sequential(
        [
            Dense(
                hidden_layer_nodes,
                activation=hidden_layer_activation,
                input_dim=input_dim,
            ),
            Dense(hidden_layer_nodes, activation=hidden_layer_activation),
            Dense(output_layer_nodes, activation=output_layer_activation),
        ]
    )
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print(model.summary())

    model.fit(
        X_train.astype("float32"),
        y_train_matrix,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
    )
    test_pred = np.argmax(model.predict(X_test.astype("float32")), axis=1)
    test_true = np.argmax(y_test_matrix, axis=1)

    print(str(round((test_pred == test_true).sum() / len(test_pred) * 100, 2)))

    return


if __name__ == "__main__":
    main()
