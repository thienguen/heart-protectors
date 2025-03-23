from setuptools import setup, find_packages

setup(
    name="heart-protectors",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.3",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "joblib>=1.1.0",
        "pytest>=7.0.0",
    ],
    author="Heart Protectors Team",
    author_email="your.email@example.com",
    description="A machine learning project for predicting heart failure risks",
    keywords="heart failure, machine learning, prediction",
    url="https://github.com/yourusername/heart-protectors",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)