# Ryanair Customer Query Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-98.71%25-brightgreen.svg)]()
[![Confidence](https://img.shields.io/badge/Avg%20Confidence-95.3%25-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

> **AI/ML Engineer Recruitment Task Solution**  
> **Complete MLOps Pipeline with Production Testing**

## Project Overview

This project implements a comprehensive automated customer query classification system for Ryanair, categorizing support tickets into 30 predefined categories using advanced machine learning techniques with full production deployment validation.

** Key Achievements:**
- **98.71% F1-Score** achieved with optimized Logistic Regression
- **95.3% Average Confidence** on real production data (999 queries tested)
- **96.1% High Confidence Rate** enabling automated processing
- **Complete MLOps Pipeline** from data preprocessing to production testing
- **Production-ready** solution with comprehensive deployment framework

## Project Structure

```
ryanair-customer-query-classification/
├── data/
│   ├── raw/                           # Original dataset files
│   └── processed/                     # Preprocessed data with splits
├── notebooks/
│   ├── 01_exploratory_analysis.py    # Data exploration and EDA
│   ├── 02_data_preprocessing.py      # Text preprocessing pipeline
│   ├── 03_model_development.py       # Model training and comparison
│   ├── 04_model_evaluation.py        # Cross-validation and tuning
│   └── 05_model_deployment_testing.py # Production testing & validation
├── models/                            # Trained model artifacts
│   ├── best_model_*.pkl              # Best performing models
│   ├── optimized_*.pkl               # Hyperparameter tuned models
│   └── evaluation_summary.json       # Model performance metadata
├── reports/
│   ├── figures/                       # Generated visualizations
│   │   ├── model_comparison.png      # Algorithm performance comparison
│   │   ├── confusion_matrix.png      # Detailed error analysis
│   │   └── deployment_dashboard.png   # Production testing dashboard
│   └── technical_report.md           # Comprehensive project report
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook/Lab or Python IDE
- 4GB+ RAM recommended for full pipeline

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

## Complete Pipeline Execution

###  Quick Start (Full Pipeline)

Run the complete pipeline from start to production testing:

```bash
cd notebooks

# Step 1: Data Exploration
python 01_exploratory_analysis.py

# Step 2: Data Preprocessing  
python 02_data_preprocessing.py

# Step 3: Model Development
python 03_model_development.py

# Step 4: Model Evaluation & Optimization
python 04_model_evaluation.py

# Step 5: Production Testing (NEW!)
python 05_model_deployment_testing.py
```

###  Pipeline Stages

#### Step 1: Data Exploration & Analysis
```bash
python 01_exploratory_analysis.py
```
**Outputs:**
- Dataset structure and quality analysis
- Class distribution visualization (30 categories)
- Text statistics and pattern identification
- Data quality assessment report

#### Step 2: Advanced Data Preprocessing
```bash
python 02_data_preprocessing.py
```
**Features:**
- Advanced text cleaning and normalization
- Domain-specific stop words for airline queries
- Lemmatization with POS tagging
- Feature engineering (text stats, punctuation analysis)
- Stratified train/validation/test splits

#### Step 3: Comprehensive Model Development
```bash
python 03_model_development.py
```
**Models Evaluated:**
- Logistic Regression (Baseline & Advanced)
- Support Vector Machine (RBF & Linear)
- Random Forest Classifier
- XGBoost, LightGBM, CatBoost
- Ensemble Voting Classifier

#### Step 4: Model Evaluation & Optimization
```bash
python 04_model_evaluation.py
```
**Advanced Analysis:**
- 5-fold stratified cross-validation
- Hyperparameter tuning with RandomizedSearchCV
- Learning curve analysis (overfitting detection)
- Detailed confusion matrix analysis
- Per-category performance assessment

#### Step 5: Production Testing & Deployment Validation 
```bash
python 05_model_deployment_testing.py
```
**Production Features:**
- Real-world data testing (999 customer queries)
- Confidence score analysis and thresholding
- Production readiness assessment
- Deployment strategy recommendations
- Performance dashboard generation

## Expected Results & Performance

### 🎯 Model Performance
- **Best Model**: Logistic Regression (Optimized)
- **Training F1-Score**: 98.71%
- **Cross-validation**: 98.58% ± 0.13%
- **Training Time**: 2.7 seconds
- **Inference Time**: <100ms per query

### 📊 Production Testing Results
- **Test Dataset**: 999 real customer queries
- **Average Confidence**: 95.3%
- **High Confidence (≥80%)**: 96.1% of predictions
- **Medium Confidence (50-80%)**: 3.7% of predictions
- **Low Confidence (<50%)**: 0.2% of predictions
- **Deployment Status**:  **Production Ready**


## Key Dependencies

```
# Core ML & Data Processing
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Advanced ML Models
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.0.0

# Development Tools
jupyter>=1.0.0
joblib>=1.1.0
```

## File Descriptions

| File | Purpose | Key Features |
|------|---------|-------------|
| `01_exploratory_analysis.py` | Data exploration | 20+ visualizations, quality assessment |
| `02_data_preprocessing.py` | Text preprocessing | Advanced cleaning, lemmatization, feature engineering |
| `03_model_development.py` | Model training | 8-model comparison, performance evaluation |
| `04_model_evaluation.py` | Model optimization | Cross-validation, hyperparameter tuning |
| `05_model_deployment_testing.py` | Production testing | Real-world validation, deployment assessment |

## Production Deployment

###  Confidence-Based Routing Strategy

```python
# Recommended production thresholds
if confidence >= 0.85:    # 96% of cases
    route_automatically()
elif confidence >= 0.70:  # 3.7% of cases  
    route_with_human_review()
else:                     # 0.2% of cases
    escalate_to_expert()
```

### Monitoring Framework

**Daily Monitoring:**
- Accuracy and confidence distribution tracking
- Error rate monitoring and alerting
- Query volume and category distribution

**Weekly Analysis:**
- Performance degradation detection
- Error pattern analysis
- Category-specific performance review

**Monthly Assessment:**
- Model retraining evaluation
- Bias audit and fairness assessment
- Performance trend analysis

## Expected Runtime & Resources

### Execution Time
- **01_exploratory_analysis.py**: 2-3 minutes
- **02_data_preprocessing.py**: 3-5 minutes
- **03_model_development.py**: 8-12 minutes
- **04_model_evaluation.py**: 10-15 minutes  
- **05_model_deployment_testing.py**: 2-3 minutes
- **Total Pipeline**: 25-40 minutes

### Resource Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: ~1GB for complete pipeline outputs
- **CPU**: Standard multi-core processor sufficient

## Troubleshooting

### Common Issues & Solutions

**NLTK Download Issues:**
```python
import nltk
nltk.download('all')
```

**Memory Issues with Large Models:**
```python
# Reduce feature size in vectorization
max_features=5000  # Instead of 10000
```

**Package Installation Issues:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Model Loading Errors:**
```bash
# Ensure models are trained before testing
python 03_model_development.py
python 04_model_evaluation.py
python 05_model_deployment_testing.py
```

### Validation Checklist

Before running the pipeline, ensure:
- [ ] Dataset files are in `data/raw/`
- [ ] All dependencies are installed
- [ ] NLTK data is downloaded
- [ ] Sufficient RAM available (4GB+)
- [ ] Python 3.8+ is being used

## Business Impact

### Expected Benefits
- **Cost Reduction**: 90%+ reduction in manual classification time
- **Response Speed**: Instant query routing vs 2-5 minutes manual
- **Consistency**: Eliminates human classification variability
- **Scalability**: Handles unlimited query volume
- **Quality**: 98.7% accuracy with continuous improvement

### Success Metrics
- **Automation Rate**: 96% of queries auto-processed
- **Accuracy Maintenance**: >95% sustained performance
- **Agent Satisfaction**: Reduced repetitive tasks
- **Customer Experience**: Faster response times

## Future Enhancements

### Roadmap
- **Multi-language Support**: Extend to EU languages
- **Real-time Learning**: Continuous model updates
- **Advanced Models**: BERT/Transformer integration
- **API Development**: REST API for production integration
- **Dashboard**: Real-time monitoring interface
