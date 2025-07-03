# ClassifyAnything - Project Scope

## Project Overview
A comprehensive, user-friendly machine learning pipeline for classification tasks, implemented entirely in Jupyter notebooks. Each step of the pipeline is a separate notebook, with data passed between steps using pickle files. The pipeline is designed for transparency, reproducibility, and accessibility, with all configuration in a top-level config.toml file.

---

## Project Structure (Notebook-Based)

```
classifyanything/
├── config.toml                           # Top-level configuration
├── notebooks/
│   ├── 01_data_ingestion.ipynb          # Load and validate data
│   ├── 02_data_exploration.ipynb        # EDA and quality assessment
│   ├── 03_preprocessing.ipynb           # Cleaning, train-test split, feature engineering
│   ├── 04_model_training.ipynb          # Train and tune multiple models
│   ├── 05_model_evaluation.ipynb        # Evaluate and compare models
│   └── 06_model_deployment.ipynb        # Deploy best model and generate reports
├── data/
│   ├── raw/                             # Original data files
│   └── processed/                       # Pickle files between notebooks
├── results/
│   ├── models/                          # Saved models
│   ├── reports/                         # HTML reports
│   └── visualizations/                  # Plots and charts
├── requirements.txt
└── README.md
```

---

## Data Flow and Best Practices
- Each notebook loads its input data from the previous step (using pickle files) and saves its output for the next step.
- **Train/test split is performed early in preprocessing** to prevent data leakage.
- **Feature engineering is performed only on the training set** and then applied to the test set.
- **No data leakage**: Test set remains completely unseen during all preprocessing steps.
- Each model is trained and tuned using only the training data and engineered features.
- Model comparison and reporting are handled in dedicated notebooks.
- All code, explanations, and results are contained within the notebooks for maximum clarity and reproducibility.

---

## Data Leakage Prevention
- **Train-test split happens early** (after basic cleaning, before feature engineering)
- **Feature engineering uses only training data** to fit transformers
- **Test data remains completely unseen** during feature engineering
- **Transformers are fitted on training data** and applied to both sets
- All feature selection and model tuning steps must use only the training data.
- The test set is used strictly for final evaluation and comparison.
- This workflow follows best practices for machine learning and ensures valid, unbiased model evaluation.

---

## Preprocessing Workflow (03_preprocessing.ipynb)
1. **Data Cleaning** (Before Split)
   - Missing value treatment
   - Outlier detection and handling
   - Duplicate removal
   - Basic data type validation

2. **Train-Test Split** (Early)
   - Split data based on config settings
   - Ensure stratification by target variable
   - Save split indices for reproducibility

3. **Feature Engineering** (After Split - Training Data Only)
   - Feature selection (correlation removal, variance selection)
   - Feature transformation (normalization, scaling)
   - Categorical encoding (if applicable)
   - All transformers fitted on training data only

4. **Data Validation**
   - Quality checks after preprocessing
   - Statistical validation
   - Ensure no data leakage

5. **Save Processed Data**
   - Clean training and test sets
   - Preprocessing pipeline objects
   - Feature metadata and documentation

---

## Configuration
- All configuration options (file paths, feature/label columns, model parameters, etc.) are stored in a single `config.toml` file at the project root.
- Each notebook loads configuration as needed.

---

## Benefits of This Approach
- **Transparency**: All steps and decisions are visible in the notebooks.
- **Reproducibility**: Data flow is explicit and controlled via pickle files.
- **Accessibility**: No need for custom Python modules or imports; everything is in the notebooks.
- **Modularity**: Easy to modify, rerun, or extend any step.
- **Best Practices**: Prevents data leakage and ensures robust model evaluation.
- **Proper ML Workflow**: Follows standard machine learning best practices.

---

## Next Steps
- Scaffold the notebook files and data folders.
- Create a config.toml template.
- Begin with 01_data_ingestion.ipynb and proceed step by step.

## Input Data Format Specification

### Standard Machine Learning Format
The pipeline uses the standard machine learning format that's compatible with scikit-learn:

**Expected Format:**
- **Samples (observations) as rows**
- **Features as columns**
- **One column contains the target labels**

### Expected CSV Format
The input CSV file should follow the standard format:

```csv
Sample_ID,Gene_001,Gene_002,Gene_003,Gene_004,Label
Sample_1,2.34,0.45,1.23,3.45,Control
Sample_2,1.87,0.67,1.45,2.98,Control
Sample_3,3.12,0.23,1.67,3.21,Disease
Sample_4,2.01,0.89,1.89,3.67,Disease
Sample_5,2.78,0.34,1.12,3.34,Control
...
```

### Format Requirements
1. **Sample ID column**: Unique identifier for each sample (optional, can be index)
2. **Feature columns**: Numeric values (gene expression levels, counts, etc.)
3. **Label column**: Target variable for classification
4. **Header row**: Column names including feature names and label column name

### Data Validation
The pipeline will automatically:
- Validate that sample IDs are unique (if provided)
- Check for missing values
- Ensure numeric data types for features
- Validate label column exists and contains valid categories
- Handle categorical labels automatically

### Configuration for Data Format
```toml
[data]
# Input data configuration
input_path = "data/dataset.csv"
label_column = "Label"  # Column name containing target labels
sample_id_column = "Sample_ID"  # Optional: column name for sample identifiers
feature_columns = []  # Empty for all columns except label and sample_id

# Data validation
validate_sample_ids = true  # Ensure unique sample identifiers
validate_labels = true  # Ensure valid label categories
```

## Core Objectives
- Handle various classification scenarios: binary, multiclass, and imbalanced datasets
- Provide comprehensive data preprocessing and quality assessment
- Implement multiple classification algorithms with automated hyperparameter tuning
- Generate detailed reports and visualizations for interpretability
- Offer both interactive (Jupyter notebooks) and automated (master script) workflows
- **Use standard machine learning format for maximum compatibility**

## Data Processing Pipeline

### Input Data
- **Standard format**: Matrix with samples as rows and features as columns (scikit-learn compatible)
- **Single file**: Contains both features and target labels
- Support for various data formats (CSV, TSV, Excel, HDF5)
- Automatic data type detection and validation
- **Format validation**: Ensure standard machine learning format compliance

### Preprocessing Steps
1. **Data Format Validation**
   - Validate standard format (samples as rows, features as columns)
   - Check sample ID uniqueness and feature name consistency
   - Verify label column exists and contains valid categories
   - Log data loading details for reproducibility

2. **Data Exploration & Quality Assessment**
   - Statistical summaries and data quality metrics
   - Missing value analysis and visualization
   - Distribution analysis and correlation matrices
   - Data shape and memory usage analysis
   - **Feature-wise and sample-wise statistics**

3. **Data Cleaning**
   - Missing value handling (imputation strategies)
   - Duplicate detection and removal
   - Data type validation and conversion
   - **Feature name standardization**

4. **Normalization/Standardization**
   - Multiple methods: Z-score, Min-Max, Robust scaling
   - Automatic method selection based on data characteristics
   - Validation of normalization effectiveness
   - **Sample-wise and feature-wise normalization options**

5. **Outlier Detection & Removal**
   - Multiple methods: IQR, Z-score, Isolation Forest
   - Configurable thresholds and strategies
   - Visualization of outlier distribution
   - **Feature-wise and sample-wise outlier detection**

6. **Feature Filtering**
   - Zero variance feature removal
   - Low variance feature filtering
   - Correlation-based feature selection
   - **Domain-specific feature filtering criteria**

7. **Label Processing**
   - Label encoding for categorical variables
   - Class balance analysis
   - Stratified sampling considerations
   - **Label validation and encoding**

## Machine Learning Workflow

### Data Splitting
- Train/Test split with configurable ratios
- Stratified splitting for imbalanced datasets
- Cross-validation setup for training set
- **Maintain feature names throughout the pipeline for interpretability**

### Feature Selection
- Univariate feature selection (chi-square, ANOVA, mutual info)
- Recursive feature elimination (RFE)
- L1-based feature selection (Lasso)
- Feature importance from tree-based models
- All feature selection performed within cross-validation to prevent data leakage
- **Return selected feature names for interpretation**

### Algorithm Selection
**Traditional Machine Learning:**
- Linear models: Logistic Regression, Linear SVM
- Tree-based: Decision Trees, Random Forest, XGBoost, LightGBM
- Kernel methods: SVM with RBF kernel
- Ensemble methods: Voting, Stacking
- Naive Bayes variants

**Deep Learning (for larger datasets):**
- Feed-forward neural networks
- Convolutional neural networks (if applicable)
- Autoencoders for feature learning
- Transfer learning options

### Hyperparameter Tuning
- Grid search and random search
- Bayesian optimization
- Cross-validation based evaluation
- Early stopping for deep learning models
- Parallel processing for faster execution

### Model Evaluation
- Multiple metrics: accuracy, precision, recall, F1, AUC-ROC, AUC-PR
- Confusion matrices and classification reports
- Learning curves and validation curves
- Statistical significance testing between algorithms
- **Gene importance analysis with biological context**

## Configuration System

### TOML Configuration File
Comprehensive configuration file with detailed comments explaining each parameter:

```toml
[data]
# Input data configuration
input_path = "data/gene_expression.csv"
labels_path = "data/sample_labels.csv"
gene_id_column = "Gene_ID"
sample_id_column = "Sample_ID"

[preprocessing]
# Data quality thresholds
missing_threshold = 0.5  # Remove genes with >50% missing values
variance_threshold = 0.01  # Remove low variance genes

# Correlation filtering options
enable_correlation_filter = true  # Set to false to skip correlation filtering
correlation_threshold = 0.95  # Remove highly correlated genes (only used if enable_correlation_filter = true)
# Alternative: set correlation_threshold = null to disable correlation filtering

# Normalization settings
normalization_method = "robust"  # zscore, minmax, robust
outlier_method = "iqr"  # iqr, zscore, isolation_forest
outlier_threshold = 1.5

# Feature selection
feature_selection_method = "mutual_info"  # chi2, anova, mutual_info, rfe
n_features = 100  # Number of genes to select

[modeling]
# Data splitting
test_size = 0.2
cv_folds = 5
random_state = 42

# Algorithms to test
algorithms = ["logistic", "random_forest", "xgboost", "svm", "neural_net"]

# Hyperparameter search
search_method = "grid"  # grid, random, bayesian
n_iterations = 100  # For random/bayesian search

[deep_learning]
# Only used if neural_net is in algorithms
hidden_layers = [64, 32]
dropout_rate = 0.3
learning_rate = 0.001
batch_size = 32
epochs = 100
early_stopping_patience = 10

[reporting]
# Output configuration
output_dir = "results"
generate_html = true
generate_pdf = false
include_plots = true
save_models = true
include_gene_names = true  # Include gene names in reports
```
