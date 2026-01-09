## Overview

The notebook demonstrates:
- Custom implementation of Naïve Bayes classifier with Laplace smoothing
- Custom implementation of Decision Tree classifier using information gain
- Evaluation on Mushroom and Congressional Voting Records datasets
- Comparison with scikit-learn's implementations
- Comprehensive performance analysis with confusion matrices

## Datasets

### 1. Mushroom Dataset
- **Size**: 8,124 samples, 22 categorical features
- **Task**: Binary classification (edible vs poisonous)
- **Source**: UCI ML Repository (ID: 73)

### 2. Congressional Voting Records Dataset
- **Size**: 435 samples, 16 categorical features
- **Task**: Binary classification (democrat vs republican)
- **Source**: UCI ML Repository (ID: 105)

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo
```

## Implementation Details

### Naïve Bayes Classifier

**Features:**
- Laplace smoothing (α=1.0) to handle zero probabilities
- Log probabilities to prevent numerical underflow
- Handles unseen feature values gracefully
- Works with categorical features

**Key Methods:**
- `fit(X, y)`: Train the classifier
- `predict_proba(X)`: Return class probabilities
- `predict(X)`: Return predicted class labels

### Decision Tree Classifier

**Features:**
- Information gain splitting criterion
- Entropy-based impurity measure
- Configurable hyperparameters:
  - `max_depth`: Maximum tree depth
  - `min_samples_split`: Minimum samples to split a node
  - `min_samples_leaf`: Minimum samples in leaf nodes

**Key Methods:**
- `fit(X, y)`: Build the decision tree
- `predict(X)`: Return predicted class labels
- `entropy(y)`: Calculate label entropy
- `information_gain(X, y, feature)`: Calculate information gain for a feature

## Data Split

- **Training**: 80% of data
- **Development**: 10% of data (for hyperparameter tuning)
- **Test**: 10% of data (for final evaluation)

## Results

### Mushroom Dataset
| Classifier | Test Accuracy |
|------------|---------------|
| Custom NB | 93.60% |
| Custom DT | 100.00% |
| Sklearn NB | 93.85% |
| Sklearn DT | 100.00% |

### Congressional Voting Records
| Classifier | Test Accuracy |
|------------|---------------|
| Custom NB | 97.73% |
| Custom DT | 95.45% |
| Sklearn NB | 97.73% |
| Sklearn DT | 100.00% |

## Usage

1. **Load and prepare data:**
```python
# Data is automatically loaded from UCI ML Repository
# Missing values are treated as a separate category
```

2. **Train custom Naïve Bayes:**
```python
nb = NaiveBayesClassifier(alpha=1.0)
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
```

3. **Train custom Decision Tree:**
```python
dt = DecisionTreeClassifier_Custom(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
```

## Notebook Structure

1. **Setup & Imports**: Load required libraries
2. **Load Datasets**: Fetch data from UCI repository
3. **Naïve Bayes Implementation**: Custom classifier class
4. **Decision Tree Implementation**: Custom tree-based classifier
5. **Evaluation Functions**: Accuracy and confusion matrix utilities
6. **Data Splitting**: Train/dev/test splits
7. **Mushroom Experiments**: Training and evaluation
8. **Voting Experiments**: Training and evaluation
9. **Summary & Visualization**: Results comparison and plots

## Key Findings

- **Decision Trees** achieved perfect accuracy on the Mushroom dataset due to clear decision boundaries
- **Naïve Bayes** performed slightly better on the Voting dataset, showing the benefit of probabilistic modeling for political voting patterns
- Custom implementations match scikit-learn performance, validating correctness
- The independence assumption of Naïve Bayes works well for categorical features

## Hyperparameter Tuning

Three Decision Tree configurations were tested:
1. No restrictions (best for Mushroom)
2. `max_depth=10, min_samples_split=5, min_samples_leaf=2`
3. `max_depth=15, min_samples_split=10, min_samples_leaf=5` (best for Voting)

## Visualization

The notebook includes:
- Bar charts comparing classifier performance
- Confusion matrices for each model
- LaTeX-formatted tables for reports

## Notes

- Missing values are explicitly handled as a separate category
- All features are categorical (no feature scaling needed)
- Random seed (42) ensures reproducible results
- Stratified splitting maintains class distribution