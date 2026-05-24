# Credit-Risk Probability of Default Project
This is a machine learning project using LendingClub loan data from 2007-2018 to predict the probability of the borrower defaulting. It is a portfolio project to develop skills suitable for quantitative finance, actuarial and risk-modelling roles.

## Overview
The goal of this project is to develop a Probability of Default (PD) model. This is a key aspect of credit risk frameworks used by banks to assess risk and ensure they conform to Basel requirements. This project covers the full process, from initial data preparation and exploration, through to model evaluation and stability testing.

## Data
The data used was found on Kaggle - All Lending Club loan data (Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club). There were initially over 2 million rows and 151 columns but this was cut to around 1.3 million rows and 38 columns after cleaning.  
The target variable was `default`, which was binary - 1 if the loan defaulted and 0 if it was fully paid. Alongside this, there were 37 variables covering borrower financial information, credit history and loan characteristics, made up of both numerical and categorical variables. All unresolved loans were excluded.

## Methodology
### 1. Data Preparation
**Filtering**

Filtered so that only completed loans were kept (status of `Fully Paid`, `Charged Off`, `Default`, `Does not meet the credit policy. Status: Fully Paid` and `Does not meet the credit policy. Status: Charged Off`). Loans that were still active were removed (`Current`, `In Grace Period`, `Late (31-120 days)`, `Late (16-30 days)` and `nan`).

**Target Variable**

The binary target variable `default` was created, with a value of 1 for `Charged Off` and `Default` statuses and then 0 for `Fully Paid`. This resulted in a heavily imbalanced dataset, with about 80.1% of loans being fully paid and the other 19.9% defaulting.

**Leakage Prevention**

Post loan variables that wouldn't have been available at the time of the application were removed to prevent data leakage. This includes hardship data and payment data. `int_rate` was kept although it could perhaps be seen as post-loan as it is set partially based on risk. The outcome of this will be discussed in a later section.

**Feature Removal**

Irrelevant columns such as URLs and other identifiers are dropped. All the joint application data is dropped as this is a small minority but would lead to a much more complicated model. Columns with large amounts of missing data are also dropped along with `installment` and `total_acc` due to their high correlations with `loan_amnt` and `open_acc` respectively.
