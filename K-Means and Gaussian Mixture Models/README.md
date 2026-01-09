# Unsupervised Learning: K-Means and Gaussian Mixture Models

This project implements K-Means and Gaussian Mixture Model (GMM) clustering algorithms from scratch and applies them to an animal measurements dataset.

## Overview

The notebook demonstrates:
- Custom implementations of K-Means and GMM clustering algorithms
- Comparison with scikit-learn implementations
- Elbow method for determining optimal cluster count
- Visualization and analysis of clustering results

## Dataset

The dataset ([hw4_dataset.csv](hw4_dataset.csv)) contains animal measurements with two features:
- **Weight** (kg)
- **Length** (cm)

The data represents 1000 animal samples that fall into distinct groups based on their physical characteristics.

## Implementations

### K-Means Clustering

Custom implementation of the K-Means algorithm featuring:
- Random centroid initialization from data points
- Iterative cluster assignment based on Euclidean distance
- Centroid updates using cluster means
- Convergence detection

Key methods:
- `fit(X)`: Train the model on data
- `predict(X)`: Assign cluster labels to new data
- `_assign_clusters(X)`: Compute nearest centroid for each point
- `_update_centroids(X)`: Update centroids as cluster means

### Gaussian Mixture Model

Custom GMM implementation using the Expectation-Maximization (EM) algorithm:
- **E-step**: Compute responsibilities (soft cluster assignments)
- **M-step**: Update means, covariances, and mixture weights
- K-Means initialization for parameters
- Log-likelihood convergence checking

Key methods:
- `fit(X)`: Train the model using EM algorithm
- `predict(X)`: Assign cluster labels based on maximum responsibility
- `_e_step(X)`: Compute posterior probabilities
- `_m_step(X, responsibilities)`: Update model parameters

## Cluster Analysis

Using the elbow method, the optimal number of clusters is determined to be **K=3**, corresponding to three distinct animal size categories:

| Cluster | Size | Weight (mean ± std) | Length (mean ± std) | Interpretation |
|---------|------|---------------------|---------------------|----------------|
| 0 | 397 | 15.08 ± 9.50 kg | 133.39 ± 14.14 cm | Small animals (cats, rabbits) |
| 1 | 203 | 8.50 ± 1.44 kg | 85.46 ± 5.05 cm | Very small animals |
| 2 | 400 | 296.30 ± 54.83 kg | 277.64 ± 22.14 cm | Large animals (horses, cattle) |

## Results

The notebook generates visualizations comparing:
1. Custom K-Means implementation
2. Custom GMM implementation
3. Scikit-learn K-Means
4. Scikit-learn GMM

All implementations produce similar clustering results, validating the custom implementations.

## Requirements

```
numpy
pandas
matplotlib
scipy
scikit-learn
seaborn
```

## Usage

Open and run [hw4.ipynb](hw4.ipynb) in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook hw4.ipynb
```

Execute cells sequentially to:
1. Load and visualize the dataset
2. Train custom and scikit-learn clustering models
3. Determine optimal K using elbow method
4. Compare clustering results
5. Analyze cluster characteristics

## Key Findings

- Both K-Means and GMM successfully identify three distinct animal size groups
- GMM provides probabilistic cluster assignments compared to K-Means' hard assignments
- Custom implementations converge quickly (4-5 iterations) and match scikit-learn results
- The elbow method clearly indicates K=3 as the optimal cluster count

## Files

- [hw4.ipynb](hw4.ipynb): Main Jupyter notebook with implementations and analysis
- `hw4_dataset.csv`: Animal measurements dataset
- [README.md](README.md): This file
