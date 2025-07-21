# Age Recognition from Speech Audio

A machine learning project focused on predicting speaker age from vocal characteristics using acoustic and linguistic features extracted from speech signals.

## Project Overview

This project implements a data science pipeline for age estimation based on speech analysis. The system analyzes acoustic properties such as pitch, formants, energy levels, and phonetic patterns to predict the chronological age of speakers from their vocal characteristics.

## Dataset Description

The dataset comprises **3,624 audio samples** with corresponding metadata:
- **Development set**: 2,933 samples (with age labels for training/validation)
- **Evaluation set**: 691 samples (for final predictions)

**Dataset Download**: The complete dataset can be downloaded from [this URL](https://drive.usercontent.google.com/download?id=1FcWrBIg63MwV26DE0882baBarakUb3HA&export=download&authuser=0)

The dataset includes comprehensive acoustic and linguistic features extracted from speech signals, along with speaker metadata such as age, gender, and ethnicity. Each sample corresponds to a spoken sentence with associated audio file and extracted features.

## Project Structure

- **`README.md`**: Project documentation and usage instructions
- **`data/`**: Dataset storage directory
  - `audios_development/`: Audio files for the development set
  - `audios_evaluation/`: Audio files for the evaluation set  
  - `development.csv`: Training data with features and age labels
  - `evaluation.csv`: Test data with features (no age labels)
  - `sample_submission.csv`: Example submission file format
- **`models/`**: Directory for saving trained model files
- **`notebooks/`**: Jupyter notebooks for analysis and experimentation
  - `data_analysis.ipynb`: Exploratory data analysis and feature visualization
  - `pipeline_prototype.ipynb`: Model development, testing, and hyperparameter tuning
- **`requirements.txt`**: List of Python package dependencies
- **`src/`**: Source code directory
  - `__init__.py`: Package initialization file
  - `feature_extractor.py`: Audio processing and feature extraction utilities
  - `train.py`: Model training pipeline and cross-validation
  - `predict.py`: Inference pipeline for generating age predictions

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train.py
```

### Prediction
```bash
python src/predict.py
```

### Exploratory Analysis
Open and run the Jupyter notebooks in the `notebooks/` directory:
- `data_analysis.ipynb`: Dataset exploration and visualization
- `pipeline_prototype.ipynb`: Model prototyping and experimentation

## Evaluation Metric

The model performance is evaluated using **Root Mean Square Error (RMSE)**:

```
RMSE = √(1/n × Σ(yi - ŷi)²)
```

Where:
- `yi`: True age values
- `ŷi`: Predicted age values
- `n`: Number of samples

RMSE provides an aggregate measure of prediction accuracy, with higher penalties for larger prediction errors.

## Submission Format

Predictions should be formatted as a CSV file with the following structure:

```csv
Id,Predicted
1,33.0
2,41.0
3,24.0
4,31.0
...
```

Each row contains:
- `Id`: Record identifier from evaluation set
- `Predicted`: Predicted age value

## Key Features

- **Comprehensive Feature Set**: Utilizes both acoustic and linguistic features for robust age prediction
- **Audio Processing**: Direct analysis of speech signals with feature extraction capabilities
- **Regression Pipeline**: End-to-end machine learning pipeline for continuous age prediction
- **Modular Design**: Clean separation of concerns with dedicated modules for training, prediction, and feature extraction

## Technical Approach

1. **Feature Engineering**: Extract and enhance acoustic/linguistic features from speech data
2. **Data Preprocessing**: Clean, normalize, and prepare features for modeling
3. **Model Selection**: Evaluate various regression algorithms for optimal performance
4. **Hyperparameter Tuning**: Optimize model parameters using cross-validation
5. **Prediction Pipeline**: Generate age predictions for evaluation dataset

## Dependencies

See `requirements.txt` for complete list of Python packages and versions required for this project.

---

*This project focuses on speech-based age recognition using machine learning techniques applied to acoustic and linguistic feature analysis.*