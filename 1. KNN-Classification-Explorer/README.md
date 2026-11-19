# KNN Classification Explorer

An exploration of how K-Nearest Neighbors (KNN) behaves when classifying data with different levels of overlap.

I built this project to experiment with KNN classification, understand how the choice of K affects predictions, and see how well the algorithm performs under different data conditions.

## Overview

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm that classifies a data point based on the labels of its nearest neighbors.

For this experiment, I generated three synthetic datasets with different levels of class overlap:

- **Easy:** Classes are clearly separated.
- **Moderate:** Classes have some overlap.
- **Difficult:** Classes have considerable overlap.

I then trained and evaluated KNN models on each dataset to investigate how classification performance changes as the overlap increases.

## What I wanted to investigate

- How does class overlap affect KNN classification performance?
- Does the optimal number of neighbors (K) change between datasets?
- Does a larger K always produce better results?
- How closely do cross-validation results reflect performance on unseen data?
- Does the model perform equally well when classifying both classes?

## Tools and Libraries

- **Python** — Main programming language
- **NumPy** — Numerical operations
- **Pandas** — Organizing and analyzing results
- **Matplotlib** — Visualizing datasets and model performance
- **Scikit-learn** — Dataset generation, preprocessing, model training, and evaluation
- **Google Colab** — Development environment

## Experiment Setup

### 1. Creating the datasets

I used Scikit-learn's `make_blobs()` function to generate three synthetic datasets.

Each dataset contains:
- 400 observations
- 2 numerical features
- 2 classes (Class A and Class B)

To control the difficulty of classification, I changed the cluster standard deviation while keeping the cluster centers and other generation settings consistent.

| Dataset | Standard Deviation | Class Overlap |
|---|---:|---|
| Easy | 1.4 | Low |
| Moderate | 2.5 | Medium |
| Difficult | 3.5 | High |

The cluster centers were set to `(1, 1)` and `(6, 6)`.

### 2. Splitting the data

Each dataset was divided into:
- 75% training data (300 observations)
- 25% testing data (100 observations)

I used stratified sampling to maintain the class proportions in both sets and a fixed random state to make the experiment reproducible.

### 3. Feature scaling

Since KNN relies on distance calculations, feature scaling is important.

I used `StandardScaler` to standardize the numerical features and combined it with the KNN classifier using Scikit-learn's `Pipeline`.

This also helped prevent data leakage during cross-validation because the scaler was fitted separately within each training fold.

### 4. Finding the best K

Rather than choosing K manually, I used `GridSearchCV` to compare several values:

```python
k_values = [1, 3, 5, 7, 11, 15, 21]
```

The experiment used five-fold cross-validation with accuracy as the scoring metric.

After selecting the best K for each dataset, I evaluated the resulting model on the corresponding test set, which was kept separate during hyperparameter tuning.

### 5. Evaluating the models

I used several metrics to examine model performance:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** How often a predicted class was correct.
- **Recall:** How many actual observations of a class were correctly identified.
- **F1-score:** A combined measure of precision and recall.
- **Confusion Matrix:** Shows correct predictions and misclassifications for each class.

## Results

### Overall Performance

| Dataset | Best K | Cross-Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Easy | 3 | 99.67% | 100.00% |
| Moderate | 3 | 91.67% | 91.00% |
| Difficult | 5 | 83.67% | 82.00% |

### Classification Report

| Dataset | Class | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| Easy | Class A | 1.00 | 1.00 | 1.00 |
| Easy | Class B | 1.00 | 1.00 | 1.00 |
| Moderate | Class A | 0.87 | 0.96 | 0.91 |
| Moderate | Class B | 0.96 | 0.86 | 0.91 |
| Difficult | Class A | 0.80 | 0.86 | 0.83 |
| Difficult | Class B | 0.85 | 0.78 | 0.81 |

### What I found

**1. Greater class overlap made classification more difficult.**

Test accuracy decreased from 100% on the Easy dataset to 82% on the Difficult dataset. This is consistent with the increased overlap between the two classes.

**2. The best K was not the same for every dataset.**

The hyperparameter search selected K = 3 for the Easy and Moderate datasets, but K = 5 for the Difficult dataset. A slightly larger neighborhood worked better for the more difficult dataset in this experiment.

**3. Cross-validation results were close to test performance.**

The cross-validation and test accuracy scores were relatively close across all three datasets. The largest difference was observed in the Difficult dataset.

**4. The model performed differently across the two classes.**

Class B had lower recall than Class A in both the Moderate and Difficult datasets. In the Difficult dataset, the model correctly identified 78% of actual Class B observations, compared with 86% for Class A.

This suggests that the classification errors were not distributed equally between the two classes in these experiments.

## Limitations

There are a few limitations worth noting:

- The experiment uses synthetic data rather than real-world datasets.
- Only two numerical features and two classes were considered.
- The search was limited to seven different K values.
- Results may change with different random seeds or train-test splits.
- Only KNN was investigated; no other classification algorithms were included for comparison.

## How to Run

### Google Colab

1. Open the notebook file in Google Colab.
2. Run the cells sequentially.

### Local Environment

Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/machine-learning-projects.git
```

Navigate to the project directory:

```bash
cd machine-learning-projects/KNN-Classification-Explorer
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `knn_classification_explorer.ipynb` and run the cells.

## Project Structure

```text
machine-learning-projects/
│
├── KNN-Classification-Explorer/
│   ├── knn_classification_explorer.ipynb
│   └── README.md
│
├── .gitignore
└── README.md
```

## Conclusion

This experiment gave me an opportunity to explore how KNN responds to changes in data distribution and class overlap.

The results showed that greater class overlap was associated with lower classification accuracy, while the optimal neighborhood size varied between datasets.

It also gave me practical experience with feature scaling, Scikit-learn pipelines, cross-validation, hyperparameter tuning, and model evaluation.

---

**Project:** KNN Classification Explorer  
**Category:** Machine Learning / Classification  
**Algorithm:** K-Nearest Neighbors (KNN)  
**Level:** Beginner–Intermediate
