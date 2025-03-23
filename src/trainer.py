import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import json
import time

# Add the current directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our model implementations
from models.model_rf import RandomForestModel
from models.model_knn import KNNModel
from models.model_svm import SVMModel
from models.model_pca import PCAModel
from preprocessing.preprocess import DataPreprocessor


class ModelTrainer:
    def __init__(self, config):
        """
        Initialize the ModelTrainer with configuration.

        Parameters:
        -----------
        config : dict
            Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.models = {}
        self.results = {}
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Create results directory if it doesn't exist
        if not os.path.exists(config['results_dir']):
            os.makedirs(config['results_dir'])

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        print("\n===== Data Loading and Preprocessing =====")

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(self.config['data_path'])

        # Load data
        data = self.preprocessor.load_data()

        if data is None:
            print("Failed to load data. Exiting.")
            sys.exit(1)

        # Explore data
        stats = self.preprocessor.explore_data()
        print(f"Dataset shape: {stats['shape']}")
        print(f"Columns: {', '.join(stats['columns'])}")

        # Check if there are missing values
        missing_values = sum(stats['missing_values'].values())
        if missing_values > 0:
            print(f"Dataset contains {missing_values} missing values.")
            print("Missing values by column:")
            for col, count in stats['missing_values'].items():
                if count > 0:
                    print(f"  {col}: {count}")
        else:
            print("No missing values found in the dataset.")

        # Get feature names (excluding target column)
        self.feature_names = [col for col in stats['columns'] if col != self.config['target_column']]

        # Preprocess data
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.preprocess_data(
            target_column=self.config['target_column'],
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

        print(f"Preprocessing completed. Training set shape: {self.X_train.shape}")

        # Save processed data if configured
        if self.config.get('save_processed_data', False):
            self.preprocessor.save_processed_data(output_dir=self.config['results_dir'])

    def train_models(self):
        """Train all the specified models."""
        if self.X_train is None or self.y_train is None:
            print("Data not preprocessed. Call load_and_preprocess_data() first.")
            return

        print("\n===== Model Training =====")

        # Train Random Forest if specified
        if 'random_forest' in self.config['models_to_train']:
            print("\n----- Training Random Forest Model -----")
            rf_model = RandomForestModel(
                n_estimators=self.config.get('rf_n_estimators', 100),
                max_depth=self.config.get('rf_max_depth', None),
                random_state=self.config['random_state']
            )

            start_time = time.time()
            rf_model.train(self.X_train, self.y_train)
            training_time = time.time() - start_time

            print(f"Random Forest training completed in {training_time:.2f} seconds.")
            self.models['random_forest'] = rf_model

            # Hyperparameter tuning if configured
            if self.config.get('perform_hyperparameter_tuning', False):
                print("\nPerforming hyperparameter tuning for Random Forest...")
                rf_model.hyperparameter_tuning(self.X_train, self.y_train)

        # Train KNN if specified
        if 'knn' in self.config['models_to_train']:
            print("\n----- Training K-Nearest Neighbors Model -----")
            knn_model = KNNModel(
                n_neighbors=self.config.get('knn_n_neighbors', 5),
                weights=self.config.get('knn_weights', 'uniform'),
                metric=self.config.get('knn_metric', 'minkowski')
            )

            start_time = time.time()
            knn_model.train(self.X_train, self.y_train)
            training_time = time.time() - start_time

            print(f"KNN training completed in {training_time:.2f} seconds.")
            self.models['knn'] = knn_model

            # Find optimal k if configured
            if self.config.get('find_optimal_k', False):
                # Split training data into train and validation
                from sklearn.model_selection import train_test_split
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    self.X_train, self.y_train, test_size=0.25, random_state=self.config['random_state']
                )

                print("\nFinding optimal k for KNN...")
                optimal_k = knn_model.find_optimal_k(
                    X_train_sub, y_train_sub, X_val, y_val,
                    k_range=range(1, self.config.get('max_k', 21))
                )

                # Retrain with optimal k
                knn_model.train(self.X_train, self.y_train)

            # Hyperparameter tuning if configured
            elif self.config.get('perform_hyperparameter_tuning', False):
                print("\nPerforming hyperparameter tuning for KNN...")
                knn_model.hyperparameter_tuning(self.X_train, self.y_train)

        # Train SVM if specified
        if 'svm' in self.config['models_to_train']:
            print("\n----- Training Support Vector Machine Model -----")
            svm_model = SVMModel(
                C=self.config.get('svm_C', 1.0),
                kernel=self.config.get('svm_kernel', 'rbf'),
                gamma=self.config.get('svm_gamma', 'scale'),
                random_state=self.config['random_state']
            )

            start_time = time.time()
            svm_model.train(self.X_train, self.y_train)
            training_time = time.time() - start_time

            print(f"SVM training completed in {training_time:.2f} seconds.")
            self.models['svm'] = svm_model

            # Hyperparameter tuning if configured
            if self.config.get('perform_hyperparameter_tuning', False):
                print("\nPerforming hyperparameter tuning for SVM...")
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
                svm_model.hyperparameter_tuning(self.X_train, self.y_train, param_grid=param_grid)

        # Apply PCA if specified
        if 'pca' in self.config['models_to_train']:
            print("\n----- Applying PCA for Dimensionality Reduction -----")
            pca_model = PCAModel(
                n_components=self.config.get('pca_n_components', None),
                random_state=self.config['random_state']
            )

            # Fit PCA
            pca_model.fit(self.X_train)
            self.models['pca'] = pca_model

            # Find optimal number of components
            print("\nFinding optimal number of components...")
            optimal_components = pca_model.find_optimal_components(
                variance_threshold=self.config.get('pca_variance_threshold', 0.95)
            )

            # Transform data using PCA
            X_train_pca = pca_model.transform(self.X_train)
            X_test_pca = pca_model.transform(self.X_test)

            print(f"Data transformed with PCA. New shape: {X_train_pca.shape}")

            # Train models on PCA-transformed data if specified
            if self.config.get('train_on_pca_data', False):
                print("\n----- Training Models on PCA-transformed Data -----")

                # Random Forest on PCA data
                if 'random_forest' in self.config['models_to_train']:
                    print("\nTraining Random Forest on PCA-transformed data...")
                    rf_pca_model = RandomForestModel(
                        n_estimators=self.config.get('rf_n_estimators', 100),
                        max_depth=self.config.get('rf_max_depth', None),
                        random_state=self.config['random_state']
                    )
                    rf_pca_model.train(X_train_pca, self.y_train)
                    self.models['random_forest_pca'] = rf_pca_model

                # KNN on PCA data
                if 'knn' in self.config['models_to_train']:
                    print("\nTraining KNN on PCA-transformed data...")
                    knn_pca_model = KNNModel(
                        n_neighbors=self.config.get('knn_n_neighbors', 5),
                        weights=self.config.get('knn_weights', 'uniform'),
                        metric=self.config.get('knn_metric', 'minkowski')
                    )
                    knn_pca_model.train(X_train_pca, self.y_train)
                    self.models['knn_pca'] = knn_pca_model

                # SVM on PCA data
                if 'svm' in self.config['models_to_train']:
                    print("\nTraining SVM on PCA-transformed data...")
                    svm_pca_model = SVMModel(
                        C=self.config.get('svm_C', 1.0),
                        kernel=self.config.get('svm_kernel', 'rbf'),
                        gamma=self.config.get('svm_gamma', 'scale'),
                        random_state=self.config['random_state']
                    )
                    svm_pca_model.train(X_train_pca, self.y_train)
                    self.models['svm_pca'] = svm_pca_model

    def evaluate_models(self):
        """Evaluate all trained models."""
        if not self.models:
            print("No models trained. Call train_models() first.")
            return

        if self.X_test is None or self.y_test is None:
            print("Test data not available. Call load_and_preprocess_data() first.")
            return

        print("\n===== Model Evaluation =====")

        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\n----- Evaluating {model_name} Model -----")

            # Skip PCA as it's not a classifier
            if model_name == 'pca':
                continue

            # For PCA-based models, use PCA-transformed test data
            if '_pca' in model_name:
                pca_model = self.models['pca']
                X_test_transformed = pca_model.transform(self.X_test)
                metrics = model.evaluate(X_test_transformed, self.y_test)
            else:
                metrics = model.evaluate(self.X_test, self.y_test)

            # Store results
            self.results[model_name] = metrics

            # Save predictions if configured
            if self.config.get('save_predictions', False):
                if '_pca' in model_name:
                    pca_model = self.models['pca']
                    X_test_transformed = pca_model.transform(self.X_test)
                    y_pred = model.predict(X_test_transformed)
                    y_prob = model.predict_proba(X_test_transformed)[:, 1]
                else:
                    y_pred = model.predict(self.X_test)
                    y_prob = model.predict_proba(self.X_test)[:, 1]

                # Create DataFrame with true labels and predictions
                pred_df = pd.DataFrame({
                    'true_label': self.y_test,
                    'predicted_label': y_pred,
                    'probability': y_prob
                })

                # Save to CSV
                pred_path = os.path.join(self.config['results_dir'], f'{model_name}_predictions.csv')
                pred_df.to_csv(pred_path, index=False)
                print(f"Predictions saved to {pred_path}")

    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        if not self.results:
            print("No evaluation results available. Call evaluate_models() first.")
            return

        print("\n===== Plotting ROC Curves =====")

        plt.figure(figsize=(10, 8))

        for model_name, metrics in self.results.items():
            # Skip PCA as it's not a classifier
            if model_name == 'pca':
                continue

            # Get predictions
            if '_pca' in model_name:
                pca_model = self.models['pca']
                X_test_transformed = pca_model.transform(self.X_test)
                y_prob = self.models[model_name].predict_proba(X_test_transformed)[:, 1]
            else:
                y_prob = self.models[model_name].predict_proba(self.X_test)[:, 1]

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)

        # Save the plot
        roc_path = os.path.join(self.config['results_dir'], 'roc_curves.png')
        plt.savefig(roc_path)
        print(f"ROC curves plot saved to {roc_path}")

        plt.close()

    def plot_precision_recall_curves(self):
        """Plot precision-recall curves for all models."""
        if not self.results:
            print("No evaluation results available. Call evaluate_models() first.")
            return

        print("\n===== Plotting Precision-Recall Curves =====")

        plt.figure(figsize=(10, 8))

        for model_name, metrics in self.results.items():
            # Skip PCA as it's not a classifier
            if model_name == 'pca':
                continue

            # Get predictions
            if '_pca' in model_name:
                pca_model = self.models['pca']
                X_test_transformed = pca_model.transform(self.X_test)
                y_prob = self.models[model_name].predict_proba(X_test_transformed)[:, 1]
            else:
                y_prob = self.models[model_name].predict_proba(self.X_test)[:, 1]

            # Compute precision-recall curve and average precision
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            ap = average_precision_score(self.y_test, y_prob)

            # Plot precision-recall curve
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)

        # Save the plot
        pr_path = os.path.join(self.config['results_dir'], 'precision_recall_curves.png')
        plt.savefig(pr_path)
        print(f"Precision-recall curves plot saved to {pr_path}")

        plt.close()

    def plot_feature_importance(self):
        """Plot feature importance for Random Forest model."""
        if 'random_forest' not in self.models:
            print("Random Forest model not trained. Call train_models() first.")
            return

        print("\n===== Plotting Feature Importance =====")

        rf_model = self.models['random_forest']

        # Get feature importance
        feature_imp = rf_model.feature_importance(self.feature_names)

        # Plot feature importance
        imp_path = os.path.join(self.config['results_dir'], 'feature_importance.png')
        rf_model.plot_feature_importance(
            feature_names=self.feature_names,
            top_n=min(10, len(self.feature_names)),
            save_path=imp_path
        )

        print(f"Feature importance plot saved to {imp_path}")

    def save_models(self):
        """Save all trained models."""
        if not self.models:
            print("No models trained. Call train_models() first.")
            return

        print("\n===== Saving Models =====")

        models_dir = os.path.join(self.config['results_dir'], 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
            try:
                model.save_model(model_path)
                print(f"{model_name} model saved to {model_path}")
            except Exception as e:
                print(f"Error saving {model_name} model: {e}")

    def save_results(self):
        """Save evaluation results to JSON."""
        if not self.results:
            print("No evaluation results available. Call evaluate_models() first.")
            return

        print("\n===== Saving Results =====")

        # Filter results to include only serializable data
        filtered_results = {}
        for model_name, metrics in self.results.items():
            filtered_metrics = {k: v for k, v in metrics.items()
                               if k not in ['classification_report']}
            filtered_results[model_name] = filtered_metrics

        # Save as JSON
        results_path = os.path.join(self.config['results_dir'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(filtered_results, f, indent=4)

        print(f"Evaluation results saved to {results_path}")

    def run_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        print("\n===== Starting Heart Failure Prediction Pipeline =====")

        # Load and preprocess data
        self.load_and_preprocess_data()

        # Train models
        self.train_models()

        # Evaluate models
        self.evaluate_models()

        # Plot ROC curves
        self.plot_roc_curves()

        # Plot precision-recall curves
        self.plot_precision_recall_curves()

        # Plot feature importance (if Random Forest was trained)
        if 'random_forest' in self.models:
            self.plot_feature_importance()

        # Save models
        self.save_models()

        # Save results
        self.save_results()

        print("\n===== Pipeline Completed =====")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Heart Failure Prediction')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--target_column', type=str, default='DEATH_EVENT',
                        help='Name of the target column')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--models', type=str, default='random_forest,knn,svm,pca',
                        help='Comma-separated list of models to train')
    parser.add_argument('--results_dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save model predictions')

    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_arguments()

    # Create configuration dictionary
    config = {
        'data_path': args.data_path,
        'target_column': args.target_column,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'models_to_train': args.models.split(','),
        'results_dir': args.results_dir,
        'perform_hyperparameter_tuning': args.tune,
        'save_predictions': args.save_predictions,
        'save_processed_data': True,
        'find_optimal_k': True,
        'train_on_pca_data': True,
        'pca_variance_threshold': 0.95
    }

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Run pipeline
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
