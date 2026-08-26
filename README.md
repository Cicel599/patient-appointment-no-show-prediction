Patient Appointment No-Show Prediction

Ulster University – MSc Computer Science with Business Development

Overview

This repository contains my MSc research project on predicting patient appointment no-shows using machine learning and explainable AI.

The project compares Logistic Regression, Random Forest, and XGBoost, and extends previous research by incorporating SHAP, LIME, McNemar's statistical testing, and a demographic fairness audit.

Research Objectives

Predict patient appointment no-shows.

Compare bagging vs boosting models.

Explain predictions using SHAP and LIME.

Evaluate fairness across demographic groups.

Dataset

Medical Appointment No-Show Dataset (Kaggle)

110,527 appointment records

Secondary dataset

Repository Structure

notebooks/

Four Colab notebooks containing the complete workflow.

figures/

Dissertation-ready figures generated during the analysis.

trained_models/

Saved Logistic Regression, Random Forest, and XGBoost models.

data_processed/

Processed data objects used by the notebooks.

Models

Model

	

Test AUC




Logistic Regression

	

0.6742




Random Forest

	

0.7265




XGBoost

	

0.7357

Key Contributions

Stratified 5-fold cross-validation

SMOTENC class balancing

SHAP explainability

LIME local explanations

McNemar's statistical significance testing

Demographic fairness audit
