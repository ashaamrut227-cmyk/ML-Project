# Predicting Personal Loan Acceptance Using Machine Learning

## 1. Project Overview

This project aims to predict whether a bank customer will accept a personal loan offer using machine learning classification algorithms. Customer information, such as income, education, family size, and other profile attributes, is used to predict loan acceptance.

The project compares baseline machine learning models with improved approaches involving class balancing, hyperparameter tuning, and Random Forest classification.

**Objective:** Build and evaluate machine learning models that identify potential personal-loan customers while addressing the class imbalance in the dataset.

**Team Members:**
- Name 1 : Asha (PES2UG24AM030)
- Name 2 : Ashmitha Sri Anand (PES2UG24AM031)


## 2. Dataset

- **Dataset:** Bank Personal Loan Modeling
- **Source:** [Kaggle – Personal Loan Modeling](https://www.kaggle.com/teertha/personal-loan-modeling)
- **Original dataset:** 5,000 customer records and 14 columns, including the ID and ZIP Code fields.
- **Target variable:** `Personal Loan`
  - `1` – Customer accepted the personal loan.
  - `0` – Customer did not accept the personal loan.

After preprocessing and removing unnecessary columns and invalid records, the dataset used in this project contains 4,987 records and 12 columns, including the target variable.

The dataset is imbalanced, with approximately 90.4% of customers not accepting the loan and 9.6% accepting it.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Machine learning classification algorithms
- Git and GitHub

## 4. Machine Learning Algorithms

### Baseline Models
1. Perceptron
2. Logistic Regression
3. K-Means clustering, adapted for classification
4. Support Vector Machine (SVM) with a linear kernel
5. Multi-Layer Perceptron (MLP)

### Improved Models
1. Perceptron with class balancing
2. Logistic Regression with class balancing
3. Linear SVM with class balancing
4. Random Forest with class balancing
5. Hyperparameter-tuned SVM
6. Hyperparameter-tuned MLP

## 5. Methodology

The project follows these steps:

1. **Data loading:** Load the customer dataset from the CSV file.
2. **Data cleaning:** Correct negative Experience values using the absolute value operation.
3. **Feature selection:** Remove the `ID` and `ZIP Code` columns.
4. **Train-test split:** Divide the dataset into 80% training data and 20% testing data using stratified sampling with `random_state=42`.
5. **Feature scaling:** Apply MinMaxScaler, fitting it only on the training data to avoid data leakage.
6. **Model training:** Train baseline and improved machine learning models.
7. **Class imbalance handling:** Use balanced class weights for selected models.
8. **Hyperparameter tuning:** Apply five-fold GridSearchCV using ROC-AUC to tune SVM and MLP.
9. **Model evaluation:** Compare models using accuracy, precision, recall, F1-score, ROC-AUC, false positives, and false negatives.

## 6. Installation and Execution

### Prerequisites
- Python 3.10 or a compatible version
- pip
- Git (optional)

### Windows Setup

Clone or download the repository and open a terminal in the project directory.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```

### Linux/macOS Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Make sure the dataset CSV is present in the `data/` directory before running the training script.

### Optional Experiments

```bash
python src/ensemble.py
python src/smote.py
python src/stacking.py
```

Run these scripts only if their corresponding implementations and dependencies are included in the repository.

## 7. Project Structure

```text
loan-prediction/
├── data/
│   └── Bank_Personal_Loan_Modelling.csv
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── evaluate.py
│   ├── train.py
│   ├── ensemble.py
│   ├── smote.py
│   └── stacking.py
├── results/
│   ├── metrics.csv
│   ├── roc_baseline.png
│   ├── roc_improved.png
│   └── confusion_matrices.png
├── requirements.txt
└── README.md
```

## 8. Results and Model Comparison

The following table summarizes the results obtained from the test dataset.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Perceptron | 91.48% | 54.78% | 65.62% | 59.72% | 0.9329 |
| Logistic Regression | 93.79% | 72.97% | 56.25% | 63.53% | 0.9610 |
| Linear SVM | 94.59% | 79.17% | 59.38% | 67.86% | 0.9611 |
| MLP | 98.70% | 93.68% | 92.71% | 93.19% | 0.9978 |
| Random Forest (balanced) | 98.50% | 87.16% | 98.96% | 92.68% | 0.9987 |
| SVM (tuned) | 95.79% | 71.09% | 94.79% | 81.25% | 0.9947 |
| MLP (tuned) | 98.50% | 90.91% | 93.75% | 92.31% | 0.9977 |

*Note: Results are from one test split. Values may vary if the data, preprocessing, or random seed changes.*

### Key Findings

- Random Forest (balanced) achieved the highest ROC-AUC of 0.9987.
- MLP achieved the highest accuracy among the listed models at 98.70%.
- Random Forest (balanced) achieved a recall of 98.96%, detecting 95 of the 96 actual loan-accepting customers in the test set.
- The tuned SVM improved recall compared with the baseline linear SVM.
- The results demonstrate why recall, precision, and ROC-AUC are useful alongside accuracy for imbalanced classification.

## 9. Visualizations

The project generates visualizations to help compare model performance:

- `roc_baseline.png` – ROC curves for baseline models.
- `roc_improved.png` – ROC curves for improved models.
- `confusion_matrices.png` – Confusion matrices showing correct and incorrect predictions.

These plots are saved in the `results/` directory after training.

## 10. Limitations

- The dataset represents a particular banking customer population and may not generalize to every bank.
- High test scores should be checked for data leakage and evaluated on additional data before practical deployment.
- K-Means is an unsupervised clustering algorithm and requires a cluster-to-class mapping to evaluate it as a classifier.
- Model predictions should support, rather than replace, responsible banking decisions.

## 11. Future Enhancements

- Experiment with SMOTE and other class imbalance techniques.
- Evaluate voting and stacking ensembles.
- Optimize classification thresholds based on business requirements.
- Use cross-validation and additional test datasets to assess generalization.
- Develop a simple web interface for interactive loan predictions.

## 12. Conclusion

This project compares several machine learning algorithms for predicting personal loan acceptance. The experiments show that model selection and class imbalance handling can substantially affect predictive performance.

The balanced Random Forest achieved the highest ROC-AUC, while the MLP achieved the highest test accuracy among the evaluated models. Further validation is necessary before using these models in real banking environments.

