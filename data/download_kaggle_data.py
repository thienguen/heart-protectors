import os
# import sys
import webbrowser
# import zipfile
# import shutil
from pathlib import Path

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# List of heart disease datasets to download
HEART_DATASETS = [
    "andrewmvd/heart-failure-clinical-data",
    "fedesoriano/heart-failure-prediction",
    "johnsmith88/heart-disease-dataset",
    "oktayrdeki/heart-disease",
]


def manual_download_instructions(dataset_name):
    """Provide instructions for manual download from Kaggle"""
    print("\n===== MANUAL DOWNLOAD INSTRUCTIONS =====")
    print(f"1. Go to: https://www.kaggle.com/datasets/{dataset_name}")
    print("2. Click the 'Download' button")
    print("3. Save the zip file to the 'data' folder")
    print("4. Extract the contents of the zip file to the 'data' folder")
    print("\nWould you like to open the dataset page in your browser? (y/n)")
    choice = input().lower().strip()
    if choice == "y":
        webbrowser.open(f"https://www.kaggle.com/datasets/{dataset_name}")


def api_download_attempt(dataset_name, path="data"):
    """Attempt to download a dataset using the Kaggle API"""
    try:
        import kaggle

        print(f"Downloading {dataset_name} using Kaggle API...")

        # Download the dataset
        kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)

        print(f"Dataset downloaded to {path}/")
        return True
    except Exception as e:
        print(f"Error using Kaggle API: {e}")
        return False


def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Create .kaggle directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)

    print("\n===== KAGGLE API SETUP =====")
    print("To use the Kaggle API, you need to provide your credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click on 'Create New API Token'")
    print("3. A kaggle.json file will be downloaded")

    print("\nWould you like to:")
    print("1. Open the Kaggle account page")
    print("2. Manually enter your Kaggle username and API key")
    print("3. Skip API setup and proceed with manual download")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        webbrowser.open("https://www.kaggle.com/account")
        print("\nAfter downloading the kaggle.json file:")
        print(f"Copy it to: {kaggle_dir}")
        input("Press Enter when you've copied the file...")

    elif choice == "2":
        username = input("Enter your Kaggle username: ")
        api_key = input("Enter your Kaggle API key: ")

        with open(kaggle_json, "w") as f:
            f.write(f'{{"username":"{username}","key":"{api_key}"}}')

        # Set permissions
        try:
            os.chmod(kaggle_json, 0o600)
        except Exception as e:
            print(f"Warning: Could not set permissions on {kaggle_json}: {e}")

    elif choice == "3":
        print("Skipping API setup...")
        return False

    else:
        print("Invalid choice. Skipping API setup...")
        return False

    return True


def download_multiple_datasets(datasets, path="data"):
    """Download multiple datasets from Kaggle"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    api_available = kaggle_json.exists()

    if not api_available:
        print("Kaggle API credentials not found!")
        print("Please enter your Kaggle credentials:")
        username = input("Username: ").strip()
        api_key = input("API Key: ").strip()

        # Create directory and save credentials
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)

        with open(kaggle_json, "w") as f:
            f.write(f'{{"username":"{username}","key":"{api_key}"}}')

        # Set permissions
        try:
            os.chmod(kaggle_json, 0o600)
        except Exception as e:
            print(f"Warning: Could not set permissions on {kaggle_json}: {e}")

    # Download each dataset
    success_count = 0
    for dataset in datasets:
        print(f"\nAttempting to download {dataset}...")
        try:
            import kaggle

            kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
            print(f"Successfully downloaded {dataset}")
            success_count += 1
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")
            manual_download_instructions(dataset)

    print(f"\nDownloaded {success_count} out of {len(datasets)} datasets")


def main():
    print("===== HEART DISEASE DATASETS DOWNLOADER =====")
    print("This script will download heart disease datasets from Kaggle")
    print("Options:")
    print("1. Download all predefined heart disease datasets")
    print("2. Specify a custom dataset to download")
    choice = input("Enter your choice (1-2): ").strip()

    if choice == "1":
        print("\nDownloading all heart disease datasets...")
        download_multiple_datasets(HEART_DATASETS)
    else:
        # Original functionality
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        api_available = kaggle_json.exists()

        if not api_available:
            print("Kaggle API credentials not found!")
            api_available = setup_kaggle_credentials()

        print("\n===== DATASET DOWNLOAD =====")
        print("Enter the Kaggle dataset you want to download")
        print("Format: username/dataset-name (e.g., 'ronitf/heart-disease-uci')")
        dataset_name = input("Dataset: ").strip()

        # Try API download if credentials are available
        if api_available:
            success = api_download_attempt(dataset_name)
            if success:
                print("\nDownload complete!")
                return

        # Fall back to manual download
        manual_download_instructions(dataset_name)


main()
