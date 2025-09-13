# Oloche's AI Oncologist: Breast Cancer Detector

**UI Display Title:** Oloche's AI Oncologist: Breast Cancer Detector

**Research Title:** Investigating the Efficacy of Machine Learning Algorithms for the Early Detection of Breast Cancer Malignancies

## Project Overview

This project implements a machine learning system for breast cancer classification using the Wisconsin Breast Cancer Dataset. The system analyzes diagnostic measurements to classify tumors as either malignant or benign with high accuracy. The model is deployed as an interactive web application using Gradio and Hugging Face Spaces.


**Webapp**: [Hugging Face Spaces Link](https://huggingface.co/spaces/eijeoloche1/Classify_tumors_as_Malignant_or_Benign)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-3.0%2B-green)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)
![License](https://img.shields.io/badge/License-Apache%25202.0-blue)

## Machine Learning Framework

- **ML Algorithms**: Implements and compares Logistic Regression, Random Forest, SVM, Gradient Boosting, and K-Nearest Neighbors
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

*oloche-ai-oncologist-breast-cancer-classifier/*

- ├── app.py (Main Gradio application)
- ├── requirements.txt (Python dependencies)
- ├── README.md (Project documentation)
- ├── models/
- │ ├── best_breast_cancer_model.pkl (Trained model
- │ └── scaler.pkl (Feature scaler)
- ├── notebooks/
- │ └── Breast_Cancer_Classification.ipynb (Complete analysis notebook)


## How to Use

- Visit the [webapp](https://huggingface.co/spaces/eijeoloche1/Classify_tumors_as_Malignant_or_Benign)
- Use the example buttons to load sample data
- Click "Analyze Data" to get predictions
- View detailed diagnostic report with confidence scores
- Please, kindly follow my advice after getting the reports

## Local Installation
1. **Clone the repository**
- git clone https://github.com/eije1/oloche-ai-oncologist-breast-cancer-classifier.git
- cd breast-cancer-classifier

2. **Install dependencies**
- pip install -r requirements.txt

3. **Run the application locally**
- python app.py
   
## Model Performance
The best performing model achieves the following results in terms of metric-score

- **Accuracy**: 97.4%
- **Precision**: 96.6%
- **Recall**: 95.8%
- **F1-Score**: 96.2%
- **AUC-ROC%**: 99.2


## Model Comparison
Comparison of the models in this project based on their accuracy

- **Random Forest (Tuned)**:	97.4%
- **SVM (Tuned)**: 97.4%
- **Gradient Boosting:** 95.6%
- **Logistic Regression:** 96.5%
- **K-Nearest Neighbors:** 95.6%
  
### Findings
1. **Top Predictive Features:**
- Worst concave points
- Worst perimeter
- Mean concave points
- Worst radius
- Worst area

2. **Class Imbalance Impact:**
Addressing class imbalance improved recall for malignant cases without sacrificing overall accuracy.
3. **Model Robustness:**
Cross-validation confirmed model stability with consistent performance across different data splits.

## Using the Web Interface
The Gradio interface provides an intuitive way to interact with the model:

**Input Methods:**
- Manually enter 30 tumor characteristic values
- Use "Load Default" for average values
- Use "Malignant Example" or "Benign Example" for sample cases
- Use "Clear All" to reset inputs

**Output Features:**
- Clear diagnosis (Benign/Malignant)
- Confidence percentage
- Probability analysis for both classes
- Risk assessment level
- Personalized medical advice

**Visual Elements:**
- Color-coded diagnosis badges
- Confidence meter visualization
- Detailed JSON results

## Hugging Face Deployment
This project is deployed on Hugging Face Spaces using the following configuration:

**Deployment Files**
- app.py: Main application file with Gradio interface
- requirements.txt: Python dependencies
- README.md: Project documentation

**Deployment Process**
- Create a new Space on Hugging Face
- Set the Space SDK to Gradio
- Upload the project files
- The Space automatically builds and deploys the application

## Technical Implementation
**Model Training**
The model was trained using a comprehensive approach:

- Multiple algorithms compared
- Hyperparameter tuning with GridSearchCV
- Class imbalance addressing
- Cross-validation for robustness

**Web Interface**
The Gradio interface includes:

- Custom CSS styling for medical aesthetic
- Input validation and error handling
- Responsive design for different devices
- Example data for quick testing

**Prediction Pipeline**
- Input validation and preprocessing
- Feature scaling using saved scaler
- Model prediction and probability calculation
- Results formatting and visualization

## Code Examples (Prediction Function)

def predict_breast_cancer(*feature_values):
    *Predict breast cancer diagnosis from input features*
    try:
        features = np.array(feature_values).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        diagnosis = "Malignant" if prediction == 0 else "Benign"
        confidence = max(prediction_proba) * 100
        
        return {
            "Diagnosis": diagnosis,
            "Confidence Level": f"{confidence:.2f}%",
            "Probability Analysis": {
                "Benign": f"{prediction_proba[1] * 100:.2f}%",
                "Malignant": f"{prediction_proba[0] * 100:.2f}%"
            },
            "Risk Assessment": "High" if diagnosis == "Malignant" else "Low",
            "Model Performance": f"{accuracy:.2%} (Validated)",
            "Medical Advice": "Consult specialist immediately" if diagnosis == "Malignant" 
                             else "Routine monitoring recommended"
        }
    except Exception as e:
        return {"Error": f"Please check all values are numbers. Error: {str(e)}"}

## Testing the Model
Test the model with these example values:

**Malignant Case Example:**

17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189

**Benign Case Example:**

13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146,
0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2,
0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259

## Recommendations
- Integration with medical imaging data
- Additional model architectures (Deep Learning)
- Patient history tracking
- Multi-language support
- Export functionality for medical reports
- API endpoints for integration with healthcare systems

## Important Disclaimer
This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult with qualified healthcare professionals for medical concerns. The model was trained on historical data and may not account for all clinical factors. Results should be interpreted by medical professionals in context with other diagnostic information.

## Researacher/Developer
- EIJE, Oloche Celestine
- Email: eijeoloche1@gmail.com

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments
- University of Wisconsin for the Breast Cancer Dataset
- Scikit-learn team for machine learning libraries
- Gradio team for the easy-to-use web interface framework
- Hugging Face for the deployment platform
- Open-source community for valuable tools and resources

## References
1. Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, 1905, 861-870.
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
3. Abid, A. (2019). Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild. ICML HILL Workshop.


