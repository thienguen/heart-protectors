import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.gridspec as gridspec

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Visualizer:
    def __init__(self, results_dir="../../results"):
        """
        Initialize the Visualizer with results directory.

        Parameters:
        -----------
        results_dir : str, default='../../results'
            Directory containing the results to visualize
        """
        self.results_dir = results_dir
        self.results = None
        self.predictions = {}
        self.models = []
        self.figures_dir = os.path.join(results_dir, "figures")

        # Create figures directory if it doesn't exist
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)

    def load_results(self):
        """Load evaluation results from JSON file."""
        results_path = os.path.join(self.results_dir, "evaluation_results.json")

        if not os.path.exists(results_path):
            print(f"Results file not found at {results_path}")
            return False

        try:
            with open(results_path, "r") as f:
                self.results = json.load(f)
                self.models = list(self.results.keys())
            print(
                f"Loaded results for {len(self.models)} models: {', '.join(self.models)}"
            )
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False

    def load_predictions(self):
        """Load model predictions from CSV files."""
        for model in self.models:
            pred_path = os.path.join(self.results_dir, f"{model}_predictions.csv")

            if os.path.exists(pred_path):
                try:
                    self.predictions[model] = pd.read_csv(pred_path)
                    print(f"Loaded predictions for {model}")
                except Exception as e:
                    print(f"Error loading predictions for {model}: {e}")
            else:
                print(f"Predictions file not found for {model}")

    def plot_model_comparison(self, metrics=None, save_path=None):
        """
        Create a bar chart comparing model performance across different metrics.

        Parameters:
        -----------
        metrics : list, default=None
            List of metrics to include in the comparison
            If None, uses ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        save_path : str, default=None
            Path to save the plot. If None, plot is saved to figures directory.
        """
        if self.results is None:
            print("Results not loaded. Call load_results() first.")
            return

        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        # Create a DataFrame for plotting
        comparison_data = []

        for model, results in self.results.items():
            for metric in metrics:
                if metric in results:
                    comparison_data.append(
                        {
                            "Model": model,
                            "Metric": metric.replace("_", " ").title(),
                            "Value": results[metric],
                        }
                    )

        if not comparison_data:
            print("No metrics data found for comparison.")
            return

        df = pd.DataFrame(comparison_data)

        # Create plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        # Create grouped bar chart
        ax = sns.barplot(x="Model", y="Value", hue="Metric", data=df)

        plt.title("Model Performance Comparison", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.legend(title="Metric", title_fontsize=12, fontsize=10, loc="best")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "model_comparison.png")

        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")
        plt.close()

    def plot_confusion_matrices(self, save_path=None):
        """
        Plot confusion matrices for all models.

        Parameters:
        -----------
        save_path : str, default=None
            Path to save the plot. If None, plot is saved to figures directory.
        """
        if self.results is None:
            print("Results not loaded. Call load_results() first.")
            return

        # Count models with confusion matrices
        models_with_cm = [
            model
            for model, results in self.results.items()
            if "confusion_matrix" in results
        ]

        if not models_with_cm:
            print("No confusion matrices found.")
            return

        # Calculate grid dimensions
        n_models = len(models_with_cm)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

        # Handle case of single subplot
        if n_models == 1:
            axes = np.array([axes])

        # Handle case of single row
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each confusion matrix
        for i, model in enumerate(models_with_cm):
            row = i // n_cols
            col = i % n_cols

            cm = np.array(self.results[model]["confusion_matrix"])

            # Calculate accuracy from confusion matrix
            accuracy = np.trace(cm) / np.sum(cm)

            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=axes[row, col], cbar=False
            )

            axes[row, col].set_title(f"{model}\nAccuracy: {accuracy:.4f}")
            axes[row, col].set_xlabel("Predicted")
            axes[row, col].set_ylabel("True")

        # Hide extra subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "confusion_matrices.png")

        plt.savefig(save_path)
        print(f"Confusion matrices plot saved to {save_path}")
        plt.close()

    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for all models.

        Parameters:
        -----------
        save_path : str, default=None
            Path to save the plot. If None, plot is saved to figures directory.
        """
        if not self.predictions:
            print("Predictions not loaded. Call load_predictions() first.")
            return

        plt.figure(figsize=(10, 8))

        for model, pred_df in self.predictions.items():
            if "true_label" in pred_df.columns and "probability" in pred_df.columns:
                try:
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(
                        pred_df["true_label"], pred_df["probability"]
                    )
                    roc_auc = auc(fpr, tpr)

                    # Plot ROC curve
                    plt.plot(fpr, tpr, lw=2, label=f"{model} (AUC = {roc_auc:.3f})")
                except Exception as e:
                    print(f"Error plotting ROC curve for {model}: {e}")
            else:
                print(f"Missing required columns in predictions file for {model}")

        # Plot the diagonal
        plt.plot([0, 1], [0, 1], "k--", lw=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save figure
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "roc_curves.png")

        plt.savefig(save_path)
        print(f"ROC curves plot saved to {save_path}")
        plt.close()

    def plot_precision_recall_curves(self, save_path=None):
        """
        Plot precision-recall curves for all models.

        Parameters:
        -----------
        save_path : str, default=None
            Path to save the plot. If None, plot is saved to figures directory.
        """
        if not self.predictions:
            print("Predictions not loaded. Call load_predictions() first.")
            return

        plt.figure(figsize=(10, 8))

        for model, pred_df in self.predictions.items():
            if "true_label" in pred_df.columns and "probability" in pred_df.columns:
                try:
                    # Calculate precision-recall curve
                    precision, recall, _ = precision_recall_curve(
                        pred_df["true_label"], pred_df["probability"]
                    )

                    # Calculate average precision
                    avg_precision = np.mean(precision)

                    # Plot precision-recall curve
                    plt.plot(
                        recall,
                        precision,
                        lw=2,
                        label=f"{model} (Avg Precision = {avg_precision:.3f})",
                    )
                except Exception as e:
                    print(f"Error plotting precision-recall curve for {model}: {e}")
            else:
                print(f"Missing required columns in predictions file for {model}")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curves", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save figure
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "precision_recall_curves.png")

        plt.savefig(save_path)
        print(f"Precision-recall curves plot saved to {save_path}")
        plt.close()

    def plot_feature_importance(self, feature_importance_path=None, save_path=None):
        """
        Plot feature importance.

        Parameters:
        -----------
        feature_importance_path : str, default=None
            Path to feature importance data. If None, tries to find in results directory.
        save_path : str, default=None
            Path to save the plot. If None, plot is saved to figures directory.
        """
        # Find feature importance file if not provided
        if feature_importance_path is None:
            feature_importance_path = os.path.join(
                self.results_dir, "feature_importance.json"
            )

        if not os.path.exists(feature_importance_path):
            print(f"Feature importance file not found at {feature_importance_path}")
            return

        try:
            # Load feature importance
            with open(feature_importance_path, "r") as f:
                feature_importance = json.load(f)

            # Convert to DataFrame
            if isinstance(feature_importance, dict):
                df = pd.DataFrame(
                    {
                        "Feature": list(feature_importance.keys()),
                        "Importance": list(feature_importance.values()),
                    }
                )
            else:
                print("Invalid feature importance format.")
                return

            # Sort by importance
            df = df.sort_values("Importance", ascending=False)

            # Plot
            plt.figure(figsize=(12, 8))

            # Create bar chart
            sns.barplot(
                x="Importance", y="Feature", data=df.head(15), palette="viridis"
            )

            plt.title("Feature Importance (Top 15)", fontsize=16)
            plt.xlabel("Importance", fontsize=14)
            plt.ylabel("Feature", fontsize=14)
            plt.tight_layout()

            # Save figure
            if save_path is None:
                save_path = os.path.join(self.figures_dir, "feature_importance.png")

            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
            plt.close()

        except Exception as e:
            print(f"Error plotting feature importance: {e}")

    def create_dashboard(self, save_path=None):
        """
        Create a comprehensive dashboard with all visualizations.

        Parameters:
        -----------
        save_path : str, default=None
            Path to save the dashboard. If None, plot is saved to figures directory.
        """
        if self.results is None:
            print("Results not loaded. Call load_results() first.")
            return

        # Create a large figure
        fig = plt.figure(figsize=(20, 16))

        # Create grid
        gs = gridspec.GridSpec(3, 2, figure=fig)

        # Model comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_comparison_on_axis(ax1)

        # Confusion matrices (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confusion_matrices_on_axis(ax2)

        # ROC curves (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_roc_curves_on_axis(ax3)

        # Precision-recall curves (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_precision_recall_curves_on_axis(ax4)

        # Feature importance (bottom full width)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_feature_importance_on_axis(ax5)

        plt.tight_layout()

        # Save dashboard
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "dashboard.png")

        plt.savefig(save_path, dpi=150)
        print(f"Dashboard saved to {save_path}")
        plt.close()

    def _plot_model_comparison_on_axis(self, ax):
        """Helper method to plot model comparison on a given axis."""
        if self.results is None:
            ax.text(
                0.5,
                0.5,
                "No results data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        # Create a DataFrame for plotting
        comparison_data = []

        for model, results in self.results.items():
            for metric in metrics:
                if metric in results:
                    comparison_data.append(
                        {
                            "Model": model,
                            "Metric": metric.replace("_", " ").title(),
                            "Value": results[metric],
                        }
                    )

        if not comparison_data:
            ax.text(
                0.5,
                0.5,
                "No metrics data found for comparison",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        df = pd.DataFrame(comparison_data)

        # Create grouped bar chart
        sns.barplot(x="Model", y="Value", hue="Metric", data=df, ax=ax)

        ax.set_title("Model Performance Comparison", fontsize=14)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.legend(title="Metric", title_fontsize=10, fontsize=8, loc="best")
        ax.tick_params(axis="x", rotation=45)

    def _plot_confusion_matrices_on_axis(self, ax):
        """Helper method to plot confusion matrices on a given axis."""
        if self.results is None:
            ax.text(
                0.5,
                0.5,
                "No results data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        # Count models with confusion matrices
        models_with_cm = [
            model
            for model, results in self.results.items()
            if "confusion_matrix" in results
        ]

        if not models_with_cm:
            ax.text(
                0.5,
                0.5,
                "No confusion matrices found",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        # Select one model for the dashboard
        model = models_with_cm[0]

        cm = np.array(self.results[model]["confusion_matrix"])

        # Calculate accuracy from confusion matrix
        accuracy = np.trace(cm) / np.sum(cm)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)

        ax.set_title(
            f"Confusion Matrix - {model}\nAccuracy: {accuracy:.4f}", fontsize=14
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)

    def _plot_roc_curves_on_axis(self, ax):
        """Helper method to plot ROC curves on a given axis."""
        if not self.predictions:
            ax.text(
                0.5,
                0.5,
                "No predictions data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        for model, pred_df in self.predictions.items():
            if "true_label" in pred_df.columns and "probability" in pred_df.columns:
                try:
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(
                        pred_df["true_label"], pred_df["probability"]
                    )
                    roc_auc = auc(fpr, tpr)

                    # Plot ROC curve
                    ax.plot(fpr, tpr, lw=2, label=f"{model} (AUC = {roc_auc:.3f})")
                except Exception as e:
                    print(f"Error plotting ROC curve for {model}: {e}")

        # Plot the diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves", fontsize=14)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_precision_recall_curves_on_axis(self, ax):
        """Helper method to plot precision-recall curves on a given axis."""
        if not self.predictions:
            ax.text(
                0.5,
                0.5,
                "No predictions data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        for model, pred_df in self.predictions.items():
            if "true_label" in pred_df.columns and "probability" in pred_df.columns:
                try:
                    # Calculate precision-recall curve
                    precision, recall, _ = precision_recall_curve(
                        pred_df["true_label"], pred_df["probability"]
                    )

                    # Calculate average precision
                    avg_precision = np.mean(precision)

                    # Plot precision-recall curve
                    ax.plot(
                        recall,
                        precision,
                        lw=2,
                        label=f"{model} (Avg Precision = {avg_precision:.3f})",
                    )
                except Exception as e:
                    print(f"Error plotting precision-recall curve for {model}: {e}")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves", fontsize=14)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance_on_axis(self, ax):
        """Helper method to plot feature importance on a given axis."""
        feature_importance_path = os.path.join(
            self.results_dir, "feature_importance.json"
        )

        if not os.path.exists(feature_importance_path):
            ax.text(
                0.5,
                0.5,
                "Feature importance data not available",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        try:
            # Load feature importance
            with open(feature_importance_path, "r") as f:
                feature_importance = json.load(f)

            # Convert to DataFrame
            if isinstance(feature_importance, dict):
                df = pd.DataFrame(
                    {
                        "Feature": list(feature_importance.keys()),
                        "Importance": list(feature_importance.values()),
                    }
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Invalid feature importance format",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                return

            # Sort by importance
            df = df.sort_values("Importance", ascending=False)

            # Create bar chart
            sns.barplot(
                x="Importance", y="Feature", data=df.head(10), palette="viridis", ax=ax
            )

            ax.set_title("Feature Importance (Top 10)", fontsize=14)
            ax.set_xlabel("Importance", fontsize=12)
            ax.set_ylabel("Feature", fontsize=12)

        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error plotting feature importance: {e}",
                ha="center",
                va="center",
                fontsize=12,
            )


def main():
    """Main function to visualize results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Heart Failure Prediction Results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../../results",
        help="Directory containing the results",
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Create a comprehensive dashboard"
    )

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = Visualizer(results_dir=args.results_dir)

    # Load results and predictions
    if visualizer.load_results():
        visualizer.load_predictions()

        # Create visualizations
        visualizer.plot_model_comparison()
        visualizer.plot_confusion_matrices()
        visualizer.plot_roc_curves()
        visualizer.plot_precision_recall_curves()
        visualizer.plot_feature_importance()

        # Create dashboard if requested
        if args.dashboard:
            visualizer.create_dashboard()


if __name__ == "__main__":
    main()
