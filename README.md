#  Ryanair Customer Query Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-98.71%25-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()


> **AI/ML Engineer Recruitment Task Solution**

## Project Overview

This project implements an automated customer query classification system for Ryanair, categorizing support tickets into 30 predefined categories using machine learning techniques.

**Key Results:**
- **98.71% F1-Score** achieved with optimized Logistic Regression
- **Comprehensive evaluation** with cross-validation and error analysis
- **Production-ready** solution with deployment recommendations

## Project Structure

```
ryanair-customer-query-classification/
├── data/
│   ├── raw/                           # Original dataset files
│   └── processed/                     # Preprocessed data
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb # Data exploration and analysis
│   ├── 02_data_preprocessing.ipynb   # Text preprocessing pipeline
│   ├── 03_model_development.ipynb    # Model training and comparison
│   └── 04_model_evaluation.ipynb     # Advanced evaluation and tuning
├── models/                            # Saved model artifacts
├── reports/
│   ├── figures/                       # Generated visualizations
│   └── final_report.md               # Detailed project report
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/[username]/ryanair-customer-query-classification.git
cd ryanair-customer-query-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## How to Run the Code

### Step 1: Data Exploration
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```
- Analyzes dataset structure and quality
- Examines class distribution and text characteristics
- Generates data insights and visualizations

### Step 2: Data Preprocessing
```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```
- Cleans and normalizes text data
- Applies lemmatization and feature engineering
- Creates train/validation splits with label encoding

### Step 3: Model Development
```bash
jupyter notebook notebooks/03_model_development.ipynb
```
- Trains and compares 8 different ML models
- Evaluates performance using multiple metrics
- Saves best performing model artifacts

### Step 4: Model Evaluation
```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```
- Performs cross-validation analysis
- Conducts hyperparameter tuning
- Provides detailed error analysis and production recommendations

## Reproducing the Results

### Complete Pipeline Execution

1. **Place dataset files** in `data/raw/` directory:
   - `customer_queries_data.csv` (training data)
   - `customer_queries_test.csv` (test data)

2. **Run notebooks in sequence:**
   ```bash
   cd notebooks
   jupyter notebook
   ```
   Execute notebooks in order: 01 → 02 → 03 → 04

3. **Expected outputs:**
   - Processed datasets in `data/processed/`
   - Model artifacts in `models/`
   - Visualizations in `reports/figures/`
   - Performance metrics and analysis results

### Expected Results
- **Final F1-Score**: 98.71%
- **Best Model**: Logistic Regression (tuned)
- **Cross-validation**: 98.58% ± 0.13%
- **Error Rate**: 1.27% on validation data

## Key Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.0.0
jupyter>=1.0.0
```

## File Descriptions

- **`01_exploratory_analysis.ipynb`**: Comprehensive EDA with 20+ visualizations
- **`02_data_preprocessing.ipynb`**: Text cleaning, lemmatization, and feature engineering
- **`03_model_development.ipynb`**: 8-model comparison framework with performance evaluation
- **`04_model_evaluation.ipynb`**: Cross-validation, hyperparameter tuning, and error analysis
- **`reports/final_report.md`**: Detailed project report with methodology and results

## Expected Runtime

- **Total execution time**: 15-20 minutes on standard hardware
- **Memory requirements**: ~2GB RAM
- **Output files**: ~500MB total size

## Troubleshooting

**NLTK Download Issues:**
```python
import nltk
nltk.download('all')
```

**Memory Issues:**
Reduce `max_features` parameter in TF-IDF vectorization (notebooks 03 and 04)

**Package Installation Issues:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Contact

For questions regarding this implementation:
- **Email**: enesozyaramiss@gmail.com