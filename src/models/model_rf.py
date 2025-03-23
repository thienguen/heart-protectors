import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import sys

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the Random Forest Classifier model.

        Parameters:
        -----------
        n_estimators : int, default=100
            The number of trees in the forest
        max_depth : int, default=None
            The maximum depth of the tree
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples
        y_train : array-like of shape (n_samples,)
            The target values
        """
        self.model.fit(X_train, y_train)
        print("Random Forest model training completed.")
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

        print(f"Random Forest evaluation metrics:")
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
        model : RandomForestClassifier
            The best model after hyperparameter tuning
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=self.params['random_state']),
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

    def feature_importance(self, feature_names=None):
        """
        Get feature importances from the trained model.

        Parameters:
        -----------
        feature_names : array-like of shape (n_features,), default=None
            Names of features

        Returns:
        --------
        importances : dict
            Dictionary mapping feature names to importance scores
        """
        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        feature_importance = dict(zip(feature_names, importances))
        sorted_importance = dict(sorted(feature_importance.items(),
                                         key=lambda x: x[1], reverse=True))

        return sorted_importance

    def plot_feature_importance(self, feature_names=None, top_n=10, save_path=None):
        """
        Plot feature importances.

        Parameters:
        -----------
        feature_names : array-like of shape (n_features,), default=None
            Names of features
        top_n : int, default=10
            Number of top features to display
        save_path : str, default=None
            Path to save the plot. If None, the plot will be displayed.
        """
        importances = self.feature_importance(feature_names)

        # Sort and get top N features
        top_features = dict(list(importances.items())[:top_n])

        plt.figure(figsize=(10, 6))
        plt.barh(list(top_features.keys())[::-1], list(top_features.values())[::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances (Random Forest)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

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
        model : RandomForestModel
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
    rf_model = RandomForestModel(n_estimators=100, max_depth=None)
    rf_model.train(X_train, y_train)

    # Evaluate
    metrics = rf_model.evaluate(X_test, y_test)

    # Feature importance
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    importances = rf_model.feature_importance(feature_names)
    print("\nFeature Importances:")
    for feature, importance in list(importances.items())[:5]:
        print(f"{feature}: {importance:.4f}")

    # Plot feature importance
    rf_model.plot_feature_importance(feature_names, top_n=10)
