import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import sys

class KNNModel:
    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski', random_state=None):
        """
        Initialize the K-Nearest Neighbors Classifier model.

        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use
        weights : str or callable, default='uniform'
            Weight function used in prediction
        metric : str or callable, default='minkowski'
            The distance metric to use
        random_state : int, default=None
            Random seed for reproducibility (if applicable)
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        self.params = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'metric': metric
        }

    def train(self, X_train, y_train):
        """
        Train the KNN model.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples
        y_train : array-like of shape (n_samples,)
            The target values
        """
        self.model.fit(X_train, y_train)
        print("KNN model training completed.")
        return self.model

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples

        Returns:
        --------
        predictions : array-like of shape (n_samples,)
            The predicted classes
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Predict class probabilities using the trained model.

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples

        Returns:
        --------
        probabilities : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples
        """
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance.

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples
        y_test : array-like of shape (n_samples,)
            The target values

        Returns:
        --------
        metrics : dict
            A dictionary containing various performance metrics
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]  # Probability of positive class

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }

        print(f"KNN evaluation metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])

        return metrics

    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using grid search.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples
        y_train : array-like of shape (n_samples,)
            The target values
        param_grid : dict, default=None
            Dictionary with parameters names as keys and lists of parameter values.
            If None, a default parameter grid will be used.
        cv : int, default=5
            Number of cross-validation folds

        Returns:
        --------
        model : KNeighborsClassifier
            The best model after hyperparameter tuning
        """
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }

        grid_search = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best ROC AUC score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        return self.model

    def find_optimal_k(self, X_train, y_train, X_val, y_val, k_range=range(1, 31), metric='accuracy'):
        """
        Find the optimal number of neighbors.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples
        y_train : array-like of shape (n_samples,)
            The target values
        X_val : array-like of shape (n_samples, n_features)
            The validation input samples
        y_val : array-like of shape (n_samples,)
            The validation target values
        k_range : range, default=range(1, 31)
            Range of k values to try
        metric : str, default='accuracy'
            Metric to optimize ('accuracy', 'f1', 'roc_auc')

        Returns:
        --------
        optimal_k : int
            The optimal number of neighbors
        """
        metric_values = []

        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if metric == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'roc_auc':
                y_prob = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_prob)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            metric_values.append(score)

        optimal_k = k_range[np.argmax(metric_values)]
        best_score = max(metric_values)

        print(f"Optimal k: {optimal_k} with {metric} = {best_score:.4f}")

        # Update model with optimal k
        self.model = KNeighborsClassifier(
            n_neighbors=optimal_k,
            weights=self.params['weights'],
            metric=self.params['metric']
        )
        self.params['n_neighbors'] = optimal_k

        # Plot k vs. metric
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, metric_values, marker='o')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel(metric.capitalize())
        plt.title(f'k vs. {metric.capitalize()}')
        plt.xticks(k_range)
        plt.grid(True)
        plt.show()

        return optimal_k

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.

        Parameters:
        -----------
        filepath : str
            Path to the saved model

        Returns:
        --------
        model : KNNModel
            The loaded model
        """
        model = cls()
        model.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               random_state=42)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Initialize and train the model
    knn_model = KNNModel(n_neighbors=5)
    knn_model.train(X_train, y_train)

    # Evaluate
    metrics = knn_model.evaluate(X_test, y_test)

    # Find optimal k
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    optimal_k = knn_model.find_optimal_k(X_train_sub, y_train_sub, X_val, y_val)

    # Train with optimal k and evaluate
    knn_model.train(X_train, y_train)
    metrics = knn_model.evaluate(X_test, y_test)
