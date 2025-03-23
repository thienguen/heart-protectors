import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import sys

class DataPreprocessor:
    def __init__(self, data_path):
        """
        Initialize the DataPreprocessor with the path to the dataset.

        Parameters:
        -----------
        data_path : str
            Path to the dataset file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load the dataset from the specified path."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def explore_data(self):
        """Perform basic data exploration and return statistics."""
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return None

        # Get basic information
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "data_types": self.data.dtypes.to_dict(),
            "descriptive_stats": self.data.describe().to_dict()
        }

        return info

    def preprocess_data(self, target_column, test_size=0.2, random_state=42):
        """
        Preprocess the dataset by handling missing values, encoding categorical variables,
        and splitting into train and test sets.

        Parameters:
        -----------
        target_column : str
            Name of the target column for prediction
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=42
            Random seed for reproducibility
        """
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return None

        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Apply preprocessing
        self.X_train_processed = preprocessor.fit_transform(self.X_train)
        self.X_test_processed = preprocessor.transform(self.X_test)

        # Store the preprocessor for future use
        self.preprocessor = preprocessor

        print(f"Data preprocessing completed. Training set shape: {self.X_train_processed.shape}")

        return self.X_train_processed, self.X_test_processed, self.y_train, self.y_test

    def save_processed_data(self, output_dir='../../results'):
        """Save the processed data to files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.X_train_processed is None or self.y_train is None:
            print("Data not processed. Call preprocess_data() first.")
            return None

        # Convert to DataFrame if needed
        if not isinstance(self.X_train_processed, pd.DataFrame):
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
                X_train_df = pd.DataFrame(self.X_train_processed, columns=feature_names)
                X_test_df = pd.DataFrame(self.X_test_processed, columns=feature_names)
            else:
                X_train_df = pd.DataFrame(self.X_train_processed)
                X_test_df = pd.DataFrame(self.X_test_processed)
        else:
            X_train_df = self.X_train_processed
            X_test_df = self.X_test_processed

        # Save to files
        X_train_df.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test_df.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)

        pd.DataFrame(self.y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        pd.DataFrame(self.y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

        print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_path = "../../data/heart_failure_dataset.csv"  # Update with your dataset name
    preprocessor = DataPreprocessor(data_path)

    # Load and explore data
    data = preprocessor.load_data()
    if data is not None:
        stats = preprocessor.explore_data()
        print("\nData Exploration:")
        print(f"Number of samples: {stats['shape'][0]}")
        print(f"Number of features: {stats['shape'][1]-1}")  # Excluding target
        print(f"Columns: {', '.join(stats['columns'])}")

        # Preprocess data
        # Assuming 'DEATH_EVENT' is the target column - adjust as needed
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(target_column='DEATH_EVENT')

        # Save processed data
        preprocessor.save_processed_data()
