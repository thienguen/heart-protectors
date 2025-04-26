import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1, L2

np.set_printoptions(threshold=np.inf)


def main():
    df = pd.read_csv("./heart.csv", index_col=False, header=0)
    y = df["Heart Disease Status"]
    X = df.drop(columns="Heart Disease Status")

    features_to_standardize = [
        "Age",
        "Blood Pressure",
        "Cholesterol Level",
        "BMI",
        "Sleep Hours",
        "Triglyceride Level",
        "Fasting Blood Sugar",
        "CRP Level",
        "Homocysteine Level",
    ]
    ordinal_features_to_encode = [
        10,
        17,
        18,
    ]
    nominal_features_to_encode = [12, 13, 14, 15, 16, 17, 18]

    nomial_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False), nominal_features_to_encode),
        ],
        remainder="passthrough",
        sparse_threshold=0,
    )
    ordinal_transformer = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(), ordinal_features_to_encode),
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

    X, X_eval, y, y_eval = train_test_split(X, y, test_size=0.2, random_state=4)
    NUM_OF_FOLDS = 20
    cv = KFold(n_splits=NUM_OF_FOLDS)
    res_train = [None] * NUM_OF_FOLDS
    res_test = [None] * NUM_OF_FOLDS
    metrics = [None] * NUM_OF_FOLDS
    metrics_to_measure = ["accuracy", "recall", "precision", "auc", "f1_score"]

    for fold_i, (train_i, test_i) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_i], X.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]
        y_test = OneHotEncoder(sparse_output=False).fit_transform(y_test.to_frame())
        y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.to_frame())

        # Z-norm standardize the training and testing data seperately to avoid weak cheating
        X_train = standardize_transformer.fit_transform(X_train)
        X_test = standardize_transformer.fit_transform(X_test)
        # encoding oridnal values
        X_train = ordinal_transformer.fit_transform(X_train)
        X_test = ordinal_transformer.fit_transform(X_test)
        # one hot encoding nominal values
        X_train = nomial_transformer.fit_transform(X_train)
        X_test = nomial_transformer.fit_transform(X_test)
        input_dim = len(X_train[0])
        hidden_layer_nodes = 512
        output_layer_nodes = 2
        hidden_layer_activation = "relu"
        output_layer_activation = "sigmoid"

        model = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(
                    hidden_layer_nodes,
                    activation=hidden_layer_activation,
                    kernel_regularizer="l1",
                    activity_regularizer="l2",
                ),
                Dropout(0.05),
                Dense(
                    hidden_layer_nodes,
                    activation=hidden_layer_activation,
                    kernel_regularizer="l2",
                ),
                Dense(output_layer_nodes, activation=output_layer_activation),
            ]
        )

        model.compile(
            optimizer=Adam(), loss="binary_crossentropy", metrics=metrics_to_measure
        )

        if fold_i == 0:
            print(model.summary())

        model.fit(
            X_train.astype("float32"),
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=128,
        )

        test_pred = np.argmax(model.predict(X_test.astype("float32")), axis=1)
        test_true = np.argmax(y_test, axis=1)
        train_pred = np.argmax(model.predict(X_train.astype("float32")), axis=1)
        train_true = np.argmax(y_train, axis=1)
        test_accuracy = round((test_pred == test_true).sum() / len(test_pred) * 100, 2)
        train_accuracy = round(
            (train_pred == train_true).sum() / len(train_pred) * 100, 2
        )
        print("Model accuracy with test data: " + str(test_accuracy) + "%")
        print("Model accuracy with training data: " + str(train_accuracy) + "%\n")

        res_test[fold_i] = test_accuracy
        res_train[fold_i] = train_accuracy
        metrics[fold_i] = model.get_metrics_result().copy()
        if fold_i == NUM_OF_FOLDS - 1:
            print(model.evaluate(X_test.astype("float32"), y_test, return_dict=True))

    print("\n--RESULTS--\n")
    print("Test data accuracy: ")
    print([float(i) for i in res_test])
    print("Train data accuracy: ")
    print([float(i) for i in res_train])
    print(
        "Average test data accuracy: "
        + str(round(sum(res_test) / len(res_test), 2))
        + "%"
    )
    print(
        "Average train data accuracy: "
        + str(round(sum(res_train) / len(res_train), 2))
        + "%"
    )

    def print_metric(metric: str):
        if metric == "f1_score":
            return
        print(
            "Average "
            + metric
            + ": "
            + str(sum([float(m[metric]) for m in metrics]) / len(metrics))
        )

    for m in metrics_to_measure:
        print_metric(m)

    print(
        "Average f1_score: "
        + str(sum([m["f1_score"][0].numpy() for m in metrics]) / len(metrics))
    )

    plt.figure(figsize=(4, 8))
    plt.plot(np.arange(1, NUM_OF_FOLDS + 1), res_test)
    plt.xticks(np.arange(1, NUM_OF_FOLDS + 1))
    plt.yticks(np.arange(70, 100))
    plt.axhline(
        y=round(sum(res_test) / len(res_test), 2),
        color="gray",
        alpha=0.3,
        linestyle="--",
    )
    plt.title("Accuracy per Fold", fontweight="bold", fontsize="x-large")
    plt.xlabel("Fold number", fontweight="semibold", fontsize="large")
    plt.ylabel("Accuracy %", fontweight="semibold", fontsize="large")
    plt.show()

    return


if __name__ == "__main__":
    main()
