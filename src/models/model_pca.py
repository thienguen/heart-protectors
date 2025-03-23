import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

class PCAModel:
    def __init__(self, n_components=None, random_state=42):
        """
        Initialize the PCA model.

        Parameters:
        -----------
        n_components : int, float or None, default=None
            Number of components to keep. If n_components is None, all components are kept.
            If n_components is float between 0 and 1, select the number of components
            such that the amount of variance that needs to be explained is greater
            than the percentage specified by n_components.
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the PCA model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples

        Returns:
        --------
        self : object
            Returns self
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Initialize PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

        # Fit PCA
        self.pca.fit(X_scaled)

        print(f"PCA model fitted with {self.pca.n_components_} components")

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_components)
            The transformed samples
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # Transform using PCA
        X_transformed = self.pca.transform(X_scaled)

        return X_transformed

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_components)
            The transformed samples
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.

        Parameters:
        -----------
        X_transformed : array-like of shape (n_samples, n_components)
            The transformed samples

        Returns:
        --------
        X_reconstructed : array-like of shape (n_samples, n_features)
            The reconstructed samples
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        # Inverse transform PCA
        X_reconstructed_scaled = self.pca.inverse_transform(X_transformed)

        # Inverse transform scaling
        X_reconstructed = self.scaler.inverse_transform(X_reconstructed_scaled)

        return X_reconstructed

    def get_explained_variance(self):
        """
        Get the explained variance ratio of each component.

        Returns:
        --------
        explained_variance_ratio : array, shape (n_components,)
            Percentage of variance explained by each of the selected components
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        return self.pca.explained_variance_ratio_

    def get_cumulative_variance(self):
        """
        Get the cumulative explained variance of components.

        Returns:
        --------
        cumulative_variance : array, shape (n_components,)
            Cumulative percentage of variance explained by components
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        return np.cumsum(self.pca.explained_variance_ratio_)

    def plot_explained_variance(self, n_components=None, save_path=None):
        """
        Plot the explained variance ratio of components.

        Parameters:
        -----------
        n_components : int, default=None
            Number of components to plot. If None, all components are plotted.
        save_path : str, default=None
            Path to save the plot. If None, the plot will be displayed.
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        if n_components is None:
            n_components = len(explained_variance)
        else:
            n_components = min(n_components, len(explained_variance))

        # Plot individual and cumulative explained variance
        plt.figure(figsize=(10, 6))

        # Bar chart for individual explained variance
        plt.bar(range(1, n_components + 1), explained_variance[:n_components],
                alpha=0.7, label='Individual explained variance')

        # Line chart for cumulative explained variance
        plt.step(range(1, n_components + 1), cumulative_variance[:n_components],
                 where='mid', label='Cumulative explained variance')

        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.xticks(range(1, n_components + 1))
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Explained variance plot saved to {save_path}")
        else:
            plt.show()

    def find_optimal_components(self, variance_threshold=0.95):
        """
        Find the optimal number of components to retain a certain amount of variance.

        Parameters:
        -----------
        variance_threshold : float, default=0.95
            The amount of variance that needs to be explained

        Returns:
        --------
        n_components : int
            The optimal number of components
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        # Find the number of components that explain at least variance_threshold
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        print(f"Optimal number of components for {variance_threshold*100}% variance: {n_components}")
        print(f"Explained variance with {n_components} components: {cumulative_variance[n_components-1]:.4f}")

        return n_components

    def plot_2d_projection(self, X, y=None, feature_names=None, components=[0, 1], save_path=None):
        """
        Plot a 2D projection of the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,), default=None
            The target values for coloring the points
        feature_names : list, default=None
            Names of features
        components : list, default=[0, 1]
            Indices of components to plot (2 components)
        save_path : str, default=None
            Path to save the plot. If None, the plot will be displayed.
        """
        if self.pca is None:
            self.fit(X)

        X_transformed = self.transform(X)

        plt.figure(figsize=(10, 8))

        if y is not None:
            # Colored scatter plot by class
            plt.scatter(X_transformed[:, components[0]], X_transformed[:, components[1]],
                      c=y, cmap='viridis', edgecolors='k', alpha=0.7)
        else:
            # Scatter plot without classes
            plt.scatter(X_transformed[:, components[0]], X_transformed[:, components[1]],
                      edgecolors='k', alpha=0.7)

        # Get feature loadings
        loadings = self.pca.components_

        # Feature names for the plot
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]

        # Calculate the explained variance for the selected components
        explained_var_0 = self.pca.explained_variance_ratio_[components[0]]
        explained_var_1 = self.pca.explained_variance_ratio_[components[1]]

        plt.xlabel(f"PC{components[0]+1} ({explained_var_0:.2f})")
        plt.ylabel(f"PC{components[1]+1} ({explained_var_1:.2f})")
        plt.title('2D PCA Projection')

        if y is not None:
            plt.colorbar(label='Class')

        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"2D projection plot saved to {save_path}")
        else:
            plt.show()

    def plot_3d_projection(self, X, y=None, feature_names=None, components=[0, 1, 2], save_path=None):
        """
        Plot a 3D projection of the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples
        y : array-like of shape (n_samples,), default=None
            The target values for coloring the points
        feature_names : list, default=None
            Names of features
        components : list, default=[0, 1, 2]
            Indices of components to plot (3 components)
        save_path : str, default=None
            Path to save the plot. If None, the plot will be displayed.
        """
        if self.pca is None:
            self.fit(X)

        X_transformed = self.transform(X)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if y is not None:
            # Colored scatter plot by class
            scatter = ax.scatter(X_transformed[:, components[0]],
                               X_transformed[:, components[1]],
                               X_transformed[:, components[2]],
                               c=y, cmap='viridis', edgecolors='k', alpha=0.7)
            plt.colorbar(scatter, label='Class')
        else:
            # Scatter plot without classes
            ax.scatter(X_transformed[:, components[0]],
                     X_transformed[:, components[1]],
                     X_transformed[:, components[2]],
                     edgecolors='k', alpha=0.7)

        # Calculate the explained variance for the selected components
        explained_var_0 = self.pca.explained_variance_ratio_[components[0]]
        explained_var_1 = self.pca.explained_variance_ratio_[components[1]]
        explained_var_2 = self.pca.explained_variance_ratio_[components[2]]

        ax.set_xlabel(f"PC{components[0]+1} ({explained_var_0:.2f})")
        ax.set_ylabel(f"PC{components[1]+1} ({explained_var_1:.2f})")
        ax.set_zlabel(f"PC{components[2]+1} ({explained_var_2:.2f})")
        plt.title('3D PCA Projection')

        if save_path:
            plt.savefig(save_path)
            print(f"3D projection plot saved to {save_path}")
        else:
            plt.show()

    def plot_feature_loadings(self, feature_names=None, components=[0, 1], save_path=None):
        """
        Plot feature loadings for the specified components.

        Parameters:
        -----------
        feature_names : list, default=None
            Names of features
        components : list, default=[0, 1]
            Indices of components to plot (2 components)
        save_path : str, default=None
            Path to save the plot. If None, the plot will be displayed.
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        loadings = self.pca.components_

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(loadings.shape[1])]

        plt.figure(figsize=(12, 10))

        # Create a loadings plot (biplot-style)
        plt.quiver(np.zeros(len(feature_names)), np.zeros(len(feature_names)),
                 loadings[components[0], :], loadings[components[1], :],
                 angles='xy', scale_units='xy', scale=1, color='r')

        # Add feature names to the plot
        for i, feature in enumerate(feature_names):
            plt.text(loadings[components[0], i] * 1.1, loadings[components[1], i] * 1.1,
                   feature, color='b')

        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # Calculate the explained variance for the selected components
        explained_var_0 = self.pca.explained_variance_ratio_[components[0]]
        explained_var_1 = self.pca.explained_variance_ratio_[components[1]]

        plt.xlabel(f"PC{components[0]+1} ({explained_var_0:.2f})")
        plt.ylabel(f"PC{components[1]+1} ({explained_var_1:.2f})")
        plt.title('PCA Feature Loadings')

        # Make the plot more square-looking
        plt.axis('equal')

        # Add a unit circle for reference
        circle = plt.Circle((0, 0), 1, fill=False, color='g', linestyle='--')
        plt.gca().add_artist(circle)

        if save_path:
            plt.savefig(save_path)
            print(f"Feature loadings plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_iris

    # Load Iris dataset as an example
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Initialize and fit PCA
    pca_model = PCAModel(n_components=None)
    pca_model.fit(X)

    # Plot explained variance
    pca_model.plot_explained_variance()

    # Find optimal number of components
    n_components = pca_model.find_optimal_components(variance_threshold=0.95)

    # Plot 2D projection
    pca_model.plot_2d_projection(X, y, feature_names=feature_names)

    # Plot 3D projection
    pca_model.plot_3d_projection(X, y, feature_names=feature_names)

    # Plot feature loadings
    pca_model.plot_feature_loadings(feature_names=feature_names)

    # Transform the data
    X_transformed = pca_model.transform(X)
    print(f"Original shape: {X.shape}, Transformed shape: {X_transformed.shape}")

    # Reconstruct the data
    X_reconstructed = pca_model.inverse_transform(X_transformed)
    print(f"Reconstruction error: {np.mean((X - X_reconstructed) ** 2):.6f}")
