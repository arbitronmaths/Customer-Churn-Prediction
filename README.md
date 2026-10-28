## **Customer Churn Prediction Using Machine Learning**

**Project Overview**

This project investigates customer churn prediction using
machine-learning classification models. It focuses on two questions: how
class weighting affects churn detection, and how changing the decision
threshold affects precision, recall, and F1-score.

The project uses the IBM Telco Customer Churn dataset and follows a
train/validation/test workflow.

**Problem Statement**
Customer churn can reduce revenue and increase customer-acquisition
costs. A churn prediction model can help identify customers who may
leave. However, churn datasets often contain fewer churners than
non-churners, so accuracy alone may not show how well a model identifies
customers who are likely to churn.

This project compares selected machine-learning models and
experimentally evaluates class weighting and decision-threshold
selection.
Research Questions
- How do selected machine-learning models compare when predicting
  customer churn?
- Does class weighting improve churn recall, and what effect does it
  have on precision and F1-score?
- How does the decision threshold affect precision, recall, and
  F1-score?
- Is the model with the highest accuracy necessarily the most useful
  for identifying churners?

**Dataset**
The project uses the IBM Telco Customer Churn dataset. The target
column indicates whether a customer churned. The notebook prepares
numerical and categorical features and excludes identifiers that should
not be used as predictive features.

Place the dataset CSV file where the notebook expects it, or update the
dataset path in the data-loading cell.
Methodology
1. Load and inspect the dataset.
2. Explore churn distribution and relevant customer features.
3. Create stratified training, validation, and test sets.
4. Compare Logistic Regression, Neural Network (MLP), Random Forest,
   and Decision Tree models.
5. Compare standard models with class-weighted variants where
   implemented.
6. Evaluate Logistic Regression across multiple decision thresholds and
   select the threshold with the highest validation F1-score.
7. Evaluate the selected model and threshold on the held-out test set.
The test set is reserved for final evaluation rather than threshold
selection.

**Evaluation Metrics**
- Accuracy: Proportion of all predictions that are correct.
- Precision (churn class): Of customers predicted to churn, the
  proportion who actually churned.
- Recall (churn class): Of customers who actually churned, the
  proportion correctly identified.
- F1-score (churn class): Harmonic mean of precision and recall.
- Confusion matrix: Counts of true negatives, false positives,
  false negatives, and true positives.
The project emphasizes churn-class precision, recall, and F1-score
because identifying potential churners is a central goal.
Main Experimental Findings
In the recorded experiment:
- Logistic Regression achieved the highest baseline validation
  F1-score among the compared baseline models (0.638).
- Class weighting increased Logistic Regression's validation churn
  recall from 0.588 to 0.821, while precision decreased from
  0.696 to 0.519.
- A decision threshold of 0.40 achieved the highest validation
  F1-score in the tested threshold grid (0.645).
- On the held-out test set, the selected Logistic Regression model at
  threshold 0.40 achieved:
  - Accuracy: 0.772
  - Churn precision: 0.560
  - Churn recall: 0.663
  - Churn F1-score: 0.607
These values describe this experimental run and should not be
interpreted as guaranteed performance on other data.

**How to Run** 

1. Install Python and Jupyter Notebook.
2. Install the common libraries used by the notebook:
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
3. Place the Telco Customer Churn CSV file in the expected location and
   check the path in the notebook.
4. Start Jupyter:
   jupyter notebook
5. Open the notebook and run the cells in order, from data loading
   through final evaluation.
Exact package requirements may vary by notebook version.

**Limitations**

- Results depend on the dataset, preprocessing, model settings, and
  random split.
- Class weighting can improve recall while increasing false positives
  and reducing precision.
- The threshold selected on validation data may not be optimal for a
  future population.
- The dataset may not represent every company's customers or churn
  behaviour.
- Further work could test additional datasets and estimate the
  business cost of false positives and false negatives.

**Future work**
- Further work could test additional datasets, evaluate ROC curves and ROC-AUC as future analysis, and estimate the business cost of false positives and false negatives.

