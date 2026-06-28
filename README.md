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

**Other Feature Updates**

- `fico_avg`: the average of `fico_range_low` and `fico_range_high`
- `credit_history_months`: the number of months between `earlies_cr_line` and `issue_d`
- `mort_acc`: missing values are filled with the median
- `emp_length`: missing values are filled with `Unknown` which is treated as a separate category. This was found to have a notably higher default rate than the others
- All other columns with missing values have the rows dropped due to the minimal amount of missing data
- `sub_grade`: natural order from A1 through to G5 so assigned a value 1 through 35
- `emp_length`: assigned a value -1 through 10 where `Unknown` is -1 and `10+ years` is 10
- `term`: mapped to integer 36 or 60 for the two unique terms
- The remaining categorical data (`purpose`, `home_ownership` and `verification_status`) is split into dummy variables due to there being no real order

### 2. Exploratory Data Analysis
**Key Findings**

- The data is heavily imbalanced - a model predicting `Fully Paid` every time would achieve 80.1% accuracy so accuracy will not be a useful measurement
- `int_rate` has the highest correlation with `default` (0.259) although it is important to be careful with this as it is set partially based on risk. Outside of `int_rate`, `fico_avg` (-0.132) and `dti` (0.109) have the highest correlation. No individual variable is strongly correlated with the target variable which confirms a model is required to capture the combined effects of multiple variables
- The default rate increases with `pub_rec_bankruptcies`, although there are a limited number of applicants with 6 or more bankruptcies where this relationship is the clearest, so this could just be noise
- `sub_grade` shows a clear monotonic increase in default rate, reflecting the validity of LendingClub's grading system
- Small business loans are the riskiest compared to wedding and car loans defaulting half as frequently
- There is no real trend in `emp_length` other than `Unknown` being noticeable higher at 25%. This confirms that it was the correct decision to assign a new value to this group instead of removing them. `10+ years` has a default rate roughly 1% lower than the others but this isn't big enough to be meaningful
- Those with a verified income status default a lot more than those who haven't verified. I expect this is because income status was only verified for lenders who seemed risky
- Defaulters with a lower income default more often. While income is also negatively biased, this is stronger in the default rate reflecting the increase in defaulters at lower incomes
- There is a clear downward trend in default rate as the number of mortgage accounts increases. These people have already been deemed trustworthy elsewhere so you would expect them to default less
- Both loan volume and default rate steadily increased from 2007-15, likely due to a loosening in lending criteria in order to aid the rapid growth

### 3. Modelling

All models used an 80/20 train/test split and the Logistic Regression models used `StandardScaler`, making sure that variables on larger scales didn't disproportionately influence the model. This was not required for the XGBoost or Random Forest models as they are tree based models that repeatedly split the data, they aren't affected by extreme values.  
Class imbalance was another are that could negatively influence the model. This was tackled by using `class_weight='balanced'` in the Balanced Logistic Regression and Random Forest models, and `scale_pos_weight` in the XGBoost.  
8 models were trained and evaluated:  
| Model | AUC | Default Recall | Default Precision | Default F1 |
| --- | --- | --- | --- | --- |
| LR Standard | 0.7096 | 0.08 | 0.53 | 0.15 |
| LR Balanced | 0.7098 | 0.64 | 0.32 | 0.43 |
| L1 LR (C=1.0) | 0.7098 | 0.64 | 0.32 | 0.43 |
| L2 LR (C=1.0) | 0.7098 | 0.64 | 0.32 | 0.43 |
| Random Forest | 0.7106 | 0.69 | 0.31 | 0.42 |
| Base XGBoost | 0.7147 | 0.68 | 0.31 | 0.43 |
| Optimised XGBoost | 0.7248 | 0.67 | 0.33 | 0.44 |
