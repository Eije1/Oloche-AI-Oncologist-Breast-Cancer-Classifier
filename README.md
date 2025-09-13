# Oloche's AI Oncologist: Breast Cancer Detector

## Project Overview

This project implements a machine learning system for breast cancer classification using the Wisconsin Breast Cancer Dataset. The system analyzes diagnostic measurements to classify tumors as either malignant or benign with high accuracy. The model is deployed as an interactive web application using Gradio and Hugging Face Spaces.

**Live Demo**: [Hugging Face Spaces Link](https://huggingface.co/spaces/eijeoloche1/Classify_tumors_as_Malignant_or_Benign)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-3.0%2B-green)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)
[![License](https://img.shields.io/badge/License-Apache%25202.0-blue)

## Machine Learning Framework

- **Multiple ML Algorithms**: Implements and compares Logistic Regression, Random Forest, SVM, Gradient Boosting, and K-Nearest Neighbors
- **Hyperparameter Tuning**: Optimizes model performance using GridSearchCV
- **Class Imbalance Handling**: Addresses dataset imbalance using class weighting
- **Interactive Web Interface**: User-friendly Gradio app deployed on Hugging Face Spaces
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Feature Importance Analysis**: Identifies most predictive tumor characteristics

## Dataset

The project uses the **Wisconsin Breast Cancer Dataset** containing:
- 569 samples (212 malignant, 357 benign)
- 30 features describing tumor characteristics
- Features include mean, standard error, and worst values of:
  - Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness
  - Compactness
  - Concavity
  - Concave points
  - Symmetry
  - Fractal dimension

##  Project Architecture

oloche-ai-oncologist-breast-cancer-classifier/

├── app.py (Main Gradio application)
├── requirements.txt (Python dependencies)
├── README.md (Project documentation)
├── models/
│ ├── best_breast_cancer_model.pkl (Trained model)
│ └── scaler.pkl (Feature scaler)
├── notebooks/
│ └── Breast_Cancer_Classification.ipynb (Complete analysis notebook)


## How to Use

**Using the Hugging Face App**
- Visit the [webapp](https://huggingface.co/spaces/eijeoloche1/Classify_tumors_as_Malignant_or_Benign)
- Use the example buttons to load sample data
- Click "Analyze Data" to get predictions
- View detailed diagnostic report with confidence scores
- Please, kindly follow my instructions after getting the reports
  
### Using the Hugging Face App

1. Visit the [live demo](https://huggingface.co/spaces/eijeoloche1/Classify_tumors_as_Malignant_or_Benign)
2. Use the example buttons to load sample data
3. Click "Analyze Data" to get predictions
4. View detailed diagnostic report with confidence scores

## Model Performance
The best performing model achieves the following results in terms of metric-score

- **Accuracy**: 97.4%
- **Precision**: 96.6%
- **Recall**: 95.8%
- **F1-Score**: 96.2%
- **AUC-ROC%**: 99.2

## Model Comparison
Model	Accuracy
Random Forest (Tuned)	97.4%
SVM (Tuned)	97.4%
Gradient Boosting	95.6%
Logistic Regression	96.5%
K-Nearest Neighbors	95.6%
  
## Findings
Top Predictive Features: The most important features for classification are:

Worst concave points

Worst perimeter

Mean concave points

Worst radius

Worst area

Class Imbalance Impact: Addressing class imbalance improved recall for malignant cases without sacrificing overall accuracy.

Model Robustness: Cross-validation confirmed model stability with consistent performance across different data splits.

🎮 Using the Web Interface
The Gradio interface provides an intuitive way to interact with the model:

Input Methods:

Manually enter 30 tumor characteristic values

Use "Load Default" for average values

Use "Malignant Example" or "Benign Example" for sample cases

Use "Clear All" to reset inputs

Output Features:

Clear diagnosis (Benign/Malignant)

Confidence percentage

Probability analysis for both classes

Risk assessment level

Personalized medical advice

Visual Elements:

Color-coded diagnosis badges

Confidence meter visualization

Detailed JSON results
