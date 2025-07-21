#  Ryanair Customer Query Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-98.71%25-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

> **AI/ML Engineer Recruitment Task Solution**  
> Automated customer query classification system achieving 98.71% F1-score

##  Project Overview

This project implements an intelligent customer query classification system for Ryanair, automatically categorizing support tickets into 30 predefined categories. The solution achieves **98.71% F1-score** using an optimized Logistic Regression model with comprehensive preprocessing and evaluation.

###  Key Results
- **98.71% F1-Score** with Logistic Regression (post-tuning)
- **1.27% error rate** on validation data
- **Production-ready** performance with monitoring strategy
- **97% automation potential** for customer service routing

###  Business Impact
- **Faster response times** through automated routing
- **Consistent categorization** across all queries
- **Significant cost reduction** in manual processing
- **Scalable solution** for growing customer volume

---

##  Project Structure

```
ryanair-customer-query-classification/
├── data/
│   ├── raw/                    # Original dataset files
│   ├── processed/              # Cleaned and preprocessed data
│   └── external/               # Additional datasets (if any)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb      # Comprehensive EDA
│   ├── 02_data_preprocessing.ipynb        # Text preprocessing pipeline
│   ├── 03_model_development.ipynb         # Model training & comparison
│   └── 04_model_evaluation.ipynb          # Advanced evaluation & tuning
├── reports/
│   ├── figures/                # Generated plots and visualizations
│   ├── final_report.md         # Comprehensive project report
│   └── eda_insights.json       # EDA findings summary
├── models/                     # Trained model artifacts
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/[your-username]/ryanair-customer-query-classification.git
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


##  Running the Analysis

### Step 1: Exploratory Data Analysis
```bash
cd notebooks
jupyter notebook 01_exploratory_analysis.ipynb
# Or use Jupyter Lab: jupyter lab 01_exploratory_analysis.ipynb
```

**What it does:**
- Analyzes dataset structure and quality
- Examines class distribution and text characteristics
- Generates visualizations for data understanding
- Creates insights summary for next steps

**Expected outputs:**
- Data quality report
- Label distribution charts
- Text length analysis
- Word frequency analysis
- Key insights JSON file

### Step 2: Data Preprocessing
```bash
jupyter notebook 02_data_preprocessing.ipynb
```

**What it does:**
- Cleans and normalizes text data
- Applies lemmatization and stop word removal
- Engineers additional text features
- Creates train/validation splits
- Saves processed data for modeling

**Expected outputs:**
- Processed training and test datasets
- Label encoder for category mapping
- Feature engineering summary
- Train/validation split files

### Step 3: Model Development
```bash
jupyter notebook 03_model_development.ipynb
```

**What it does:**
- Trains 8 different machine learning models
- Compares performance across algorithms
- Generates comprehensive evaluation metrics
- Creates model comparison visualizations
- Saves best performing models

**Expected outputs:**
- Model performance comparison
- Confusion matrix analysis
- Best model artifacts (`.pkl` files)
- Training time benchmarks

### Step 4: Model Evaluation & Optimization
```bash
jupyter notebook 04_model_evaluation.ipynb
```

**What it does:**
- Performs cross-validation analysis
- Conducts hyperparameter tuning
- Analyzes learning curves and overfitting
- Provides detailed error analysis
- Generates production recommendations

**Expected outputs:**
- Cross-validation results
- Hyperparameter tuning improvements
- Learning curve analysis
- Error pattern identification
- Production deployment strategy

---

##  Results Summary

### Model Performance

| Model | F1-Score | Accuracy | Training Time | Status |
|-------|----------|----------|---------------|---------|
| **Logistic Regression (Tuned)** | **98.71%** | **98.71%** | 2.7s | 🏆 **Best** |
| SVM | 98.58% | 98.58% | 96.3s | Excellent |
| Random Forest | 98.42% | 98.42% | 8.5s | Very Good |
| XGBoost | 98.01% | 98.01% | 107.5s | Good |
| Ensemble Voting | 98.48% | 98.48% | 426.0s | Excellent |

### Cross-Validation Stability
- **Logistic Regression**: 98.58% ± 0.13% (highly stable)
- **Minimal overfitting**: Training-validation gap of only 0.009
- **Consistent performance**: All models show <0.3% variance

### Error Analysis
- **Total error rate**: 1.27% (51 errors out of 4,000 validation samples)
- **Logical error patterns**: Semantically similar categories (e.g., Frequent Flyer ↔ Loyalty Programs)
- **No systematic bias**: Errors distributed across categories

---

##  Technical Details

### Data Processing Pipeline
1. **Text Cleaning**: URL removal, normalization, special character handling
2. **Tokenization**: NLTK-based word tokenization
3. **Lemmatization**: POS-aware lemmatization for semantic preservation
4. **Feature Engineering**: Text statistics, punctuation analysis, keyword detection
5. **Vectorization**: TF-IDF with optimized parameters (max_features=10000, ngrams=(1,2))

### Model Architecture
- **Primary Model**: Logistic Regression with L2 regularization
- **Hyperparameters**: C=10.0, solver='liblinear', penalty='l2'
- **Input Features**: TF-IDF vectors (10,000 dimensions)
- **Output**: 30 category probabilities with confidence scores

### Evaluation Strategy
- **Primary Metric**: Weighted F1-Score (handles class imbalance)
- **Cross-Validation**: 5-fold stratified CV for robust estimation
- **Hyperparameter Tuning**: RandomizedSearchCV with 20 iterations
- **Error Analysis**: Confusion matrix and misclassification pattern analysis

---

##  Key Findings

###  Best Practices Discovered
1. **Simple models can outperform complex ones** on well-preprocessed text data
2. **Lemmatization significantly improves** performance over basic cleaning
3. **Hyperparameter tuning provides** meaningful improvements (0.13% gain)
4. **Cross-validation is essential** for reliable performance estimation

###  Business Insights
1. **Payment Issues** and **Flight Changes** are most common categories
2. **Perfect classification** achieved for Business Travel, Group Bookings
3. **Semantic similarities** cause most classification errors
4. **Confidence thresholds** can further reduce error impact

###  Technical Insights
1. **TF-IDF remains highly effective** for text classification
2. **Balanced preprocessing** more important than complex models
3. **Feature engineering** provides marginal but valuable improvements
4. **Ensemble methods** don't always improve well-tuned simple models

---

##  Production Deployment

### Recommended Architecture
```
Customer Query → API Gateway → ML Service → Confidence Check → Route Decision
                                     ↓
                             Model Registry ← Monitoring System
```

### Deployment Strategy
- **Confidence Threshold**: 95% for automatic routing
- **Human Review**: Queries below 95% confidence
- **API Response Time**: <100ms per query
- **Monitoring**: Real-time performance tracking
- **Retraining**: Monthly model updates

### Scaling Considerations
- **Containerization**: Docker for consistent deployment
- **Orchestration**: Kubernetes for auto-scaling
- **Load Balancing**: Handle 1000+ queries/minute
- **Monitoring**: Comprehensive performance and drift detection

---

##  File Descriptions

### Core Analysis Files
- **`01_exploratory_analysis.ipynb`**: Complete EDA with 20+ visualizations
- **`02_data_preprocessing.ipynb`**: Advanced text preprocessing pipeline
- **`03_model_development.ipynb`**: 8-model comparison framework
- **`04_model_evaluation.ipynb`**: Cross-validation and optimization

### Generated Artifacts
- **`models/best_model_*.pkl`**: Trained model artifacts
- **`models/evaluation_summary.json`**: Comprehensive results
- **`reports/figures/`**: All generated visualizations
- **`data/processed/`**: Clean, ready-to-use datasets

### Configuration Files
- **`requirements.txt`**: All Python dependencies
- **`.gitignore`**: Optimized for ML projects
- **`reports/final_report.md`**: Detailed project report

---

##  Reproducibility

### Exact Reproduction Steps

1. **Environment Setup** (5 minutes)
```bash
git clone [repository-url]
cd ryanair-customer-query-classification
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Data Preparation** (2 minutes)
```bash
# Place CSV files in data/raw/
# customer_queries_data.csv
# customer_queries_test.csv
```

3. **Run Complete Pipeline** (15-20 minutes)
```bash
cd notebooks
jupyter notebook
# Open and run each notebook in order:
# 01_exploratory_analysis.ipynb      # ~3 minutes
# 02_data_preprocessing.ipynb         # ~2 minutes  
# 03_model_development.ipynb          # ~8 minutes
# 04_model_evaluation.ipynb           # ~5 minutes
```

4. **Review Results**
```bash
# Check generated files
ls models/                    # Model artifacts
ls reports/figures/          # Visualizations
cat models/evaluation_summary.json  # Final results
```

### Expected Timeline
- **Total runtime**: 15-20 minutes on modern hardware
- **Peak memory usage**: ~2GB RAM
- **Disk space**: ~500MB for all artifacts

---

##  Troubleshooting

### Common Issues

**1. NLTK Download Errors**
```python
import nltk
nltk.download('all')  # Download all NLTK data
```

**2. Memory Issues**
```python
# Reduce TF-IDF max_features in preprocessing
max_features=5000  # Instead of 10000
```

**3. Package Conflicts**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**4. Virtual Environment Issues**
```bash
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Performance Optimization

**For Faster Training:**
- Reduce `n_iter` in hyperparameter tuning (line 342 in `04_model_evaluation.py`)
- Use fewer CV folds (change `cv=5` to `cv=3`)
- Limit TF-IDF features (`max_features=5000`)


##  Dependencies

### Core Libraries
```
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning
nltk>=3.6.0            # Natural language processing
```

### Visualization
```
matplotlib>=3.5.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
plotly>=5.0.0          # Interactive charts
wordcloud>=1.8.1       # Word cloud generation
```

### Machine Learning Extensions
```
xgboost>=1.6.0         # Gradient boosting
lightgbm>=3.3.0        # Gradient boosting
catboost>=1.0.0        # Gradient boosting
```

### Development Tools
```
jupyter>=1.0.0         # Notebook environment
joblib>=1.1.0          # Model serialization
tqdm>=4.64.0           # Progress bars
```

---

