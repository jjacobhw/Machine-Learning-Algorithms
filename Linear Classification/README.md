## Overview
This project implements and compares three linear classification algorithms:
1. **Perceptron** (custom implementation from scratch)
2. **Linear SVM** (using scikit-learn)
3. **Logistic Regression** (using scikit-learn)

The classifiers are evaluated on multiple datasets:
- **AND Gate**: Simple logic gate test for Perceptron validation
- **Synthetic Data**: Generated linearly separable data for controlled testing
- **Spambase**: Email spam classification (fetched from UCI ML Repository)
- **UDHR**: Language identification (English vs. Dutch)

## Project Structure
```
Linear Classification/
├── README.md                      # This file
├── script.py                      # Main implementation
├── report.pdf                     # Written report
└── universal-declaration/
    ├── english.txt               # English UDHR text
    └── dutch.txt                 # Dutch UDHR text
```

## Features

### Custom Implementations
- **Perceptron Classifier**: Full implementation with configurable learning rate and iterations
- **Confusion Matrix**: Custom implementation for binary classification
- **Evaluation Metrics**: Custom calculation of accuracy, precision, recall, and F1 score
- **Evaluation Reporting**: Formatted output with confusion matrices

### Spam Classification
Uses 5 selected features from the Spambase dataset:
- `word_freq_free`
- `word_freq_credit`
- `word_freq_money`
- `capital_run_length_average`
- `word_freq_receive`

Data split: 70% train / 10% dev / 20% test

### Language Classification
Uses combined feature extraction:
- **Character n-grams**: 2-4 character sequences (1000 features)
- **Word n-grams**: 1-2 word sequences (500 features)
- **Total**: 1500 combined features

Custom development and test sets with 20 sentences each per language.

## Requirements
```bash
pip install numpy pandas scikit-learn matplotlib seaborn ucimlrepo
```

## Running the Code

Run all experiments:
```bash
python script.py
```

This will execute in order:
1. Language classification (English vs Dutch)
2. Spam classification
3. Perceptron evaluation on synthetic data
4. Perceptron test on AND gate

Each experiment outputs:
- Development set metrics (where applicable)
- Test set metrics
- Confusion matrices
- Model comparison tables
