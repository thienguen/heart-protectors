import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import sys

class SVMModel:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=42):
        """
        Initialize the Support Vector Machine Classifier model.

        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter
        kernel : str, default='rbf'
            Kernel type to be used in the algorithm
        gamma : str or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=random_state
        )
        self.params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'random_state': random_state
        }

    def train(self, X_train, y_train):
        """
        Train the SVM model.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples
        y_train : array-like of shape (n_samples,)
            The target values
        """
        self.model.fit(X_train, y_train)
        print("SVM model training completed.")
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

        print(f"SVM evaluation metrics:")
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
        model : SVC
            The best model after hyperparameter tuning
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
            }

        grid_search = GridSearchCV(
            estimator=SVC(probability=True, random_state=self.params['random_state']),
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

    def plot_decision_boundary(self, X, y, feature_names=None, feature_indices=[0, 1]):
        """
        Plot decision boundary for 2D data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,)
            The target values
        feature_names : list, default=None
            Names of features
        feature_indices : list, default=[0, 1]
            Indices of features to plot (only 2 features can be used)
        """
        if len(feature_indices) != 2:
            raise ValueError("feature_indices must contain exactly 2 indices")

        # Extract the two features for plotting
        X_plot = X[:, feature_indices]

        h = 0.02  # Step size in the mesh

        # Create mesh grid
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Create meshgrid with all features
        if X.shape[1] > 2:
            meshgrid = np.zeros((xx.ravel().shape[0], X.shape[1]))
            # Fill in with mean values for non-plotted features
            for i in range(X.shape[1]):
                if i in feature_indices:
                    idx = feature_indices.index(i)
                    if idx == 0:
                        meshgrid[:, i] = xx.ravel()
                    else:
                        meshgrid[:, i] = yy.ravel()
                else:
                    meshgrid[:, i] = np.mean(X[:, i])
        else:
            meshgrid = np.c_[xx.ravel(), yy.ravel()]

        # Predict class labels for mesh grid points
        Z = self.model.predict(meshgrid)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and training points
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)

        if feature_names is not None:
            plt.xlabel(feature_names[feature_indices[0]])
            plt.ylabel(feature_names[feature_indices[1]])
        else:
            plt.xlabel(f'Feature {feature_indices[0]}')
            plt.ylabel(f'Feature {feature_indices[1]}')

        plt.title('SVM Decision Boundary')
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
        model : SVMModel
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
    svm_model = SVMModel(C=1.0, kernel='rbf')
    svm_model.train(X_train, y_train)

    # Evaluate
    metrics = svm_model.evaluate(X_test, y_test)

    # Hyperparameter tuning (limited to a small grid for demonstration)
    small_param_grid = {
        'C': [0.1, 1.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_model.hyperparameter_tuning(X_train, y_train, param_grid=small_param_grid)

    # Evaluate after tuning
    metrics_tuned = svm_model.evaluate(X_test, y_test)

    # Plot decision boundary (using first two features)
    try:
        svm_model.plot_decision_boundary(X, y, feature_indices=[0, 1])
    except Exception as e:
        print(f"Could not plot decision boundary: {e}")
