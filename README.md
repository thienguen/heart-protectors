# Heart Failure Risk Prediction

A machine learning project for predicting heart failure risks based on patient health indicators.

## 📋 Project Overview

This project implements various machine learning models to predict heart failure risks using medical records and health indicators. We analyze patient data using different algorithms and evaluate their performance.

## 📊 Dataset Information

We use the following heart disease datasets from Kaggle:

1. [Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) - 299 patients with 13 features for predicting mortality
2. [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) - 918 patients with 11 features for heart disease prediction
3. [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) - Larger dataset with additional clinical features
4. [Heart Disease UCI](https://www.kaggle.com/datasets/oktayrdeki/heart-disease) - Classic UCI heart disease dataset

## 📁 Project Structure

```
heart-protectors/
├── data/                           # Dataset directory
│   ├── heart_failure_clinical_records_dataset.csv
│   ├── heart.csv
│   └── heart_disease.csv
├── research/                       # Jupyter notebooks for analysis
│   └── heart_disease_eda.ipynb     # Exploratory Data Analysis
├── src/                            # Source code
│   ├── preprocessing/              # Data preprocessing scripts
│   │   └── preprocess.py           # Handles data cleaning and preparation
│   ├── models/                     # ML model implementations
│   │   ├── model_rf.py             # Random Forest implementation
│   │   ├── model_knn.py            # K-Nearest Neighbors implementation
│   │   ├── model_svm.py            # Support Vector Machine implementation
│   │   └── model_pca.py            # PCA implementation
│   ├── visualization/              # Visualization scripts
│   │   └── visualize.py            # Model performance visualization
│   └── trainer.py                  # Model training and evaluation
├── results/                        # Model evaluation results
├── download_kaggle_data.py         # Script to download datasets from Kaggle
├── README.md                       # Project documentation
└── requirements.txt                # Project dependencies
```

## 🤖 Models Implemented

- **🌳 Random Forest**: Ensemble learning method for classification
- **🏘️ K-Nearest Neighbors (KNN)**: Instance-based learning for classification
- **🔍 Support Vector Machine (SVM)**: Effective for high-dimensional spaces
- **📉 Principal Component Analysis (PCA)**: Dimensionality reduction technique

## ✅ Implementation Checklist

### 🚀 Setup

- [x] Initialize Git repository
- [x] Create project structure
- [x] Create requirements.txt

### 🧹 Data Preparation

- [ ] Load dataset
- [ ] Handle missing values
- [ ] Encode categorical variables
- [ ] Normalize numerical features
- [ ] Split data into training/validation/test sets

### 💡 Model Implementation

- [ ] Implement Random Forest model
- [ ] Implement KNN model
- [ ] Implement SVM model
- [ ] Apply PCA for dimensionality reduction

### 📈 Model Evaluation

- [ ] Train and evaluate Random Forest
- [ ] Train and evaluate KNN
- [ ] Train and evaluate SVM
- [ ] Compare model performances
- [ ] Visualize results

### 🔬 Advanced Topics (If Time Permits)

- [ ] Implement neural network models
- [ ] Create web interface for model demonstration
- [ ] Perform hyperparameter optimization

## 🛠️ Setup Instructions

1. Clone the repository:

```bash
git clone <repository-url>
cd heart-protectors
```

2. Create and activate a virtual environment:

```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download datasets:

```bash
# Option 1: Run the download script (requires Kaggle API credentials)
python download_kaggle_data.py

# Option 2: Download manually from Kaggle links above and place files in the data/ directory
```

5. Run exploratory data analysis:

```bash
jupyter notebook research/heart_disease_eda.ipynb
```

6. Run preprocessing:

```bash
python src/preprocessing/preprocess.py
```

7. Train and evaluate models:

```bash
python src/trainer.py
```

## 🔑 Setting up Kaggle API Credentials

To use the download script with Kaggle API:

1. Create a Kaggle account if you don't have one
2. Go to https://www.kaggle.com/account
3. Click on "Create New API Token"
4. Save the kaggle.json file to ~/.kaggle/ directory
   - On Windows: C:\Users\<username>\.kaggle\kaggle.json
   - On Linux/Mac: ~/.kaggle/kaggle.json
5. Ensure permissions are set correctly (chmod 600 ~/.kaggle/kaggle.json on Linux/Mac)

## 🔮 Future Enhancements

- 🌐 Web interface for interactive predictions
- 🧠 Implementation of deep learning models
- 📱 Mobile application for quick heart health assessments
