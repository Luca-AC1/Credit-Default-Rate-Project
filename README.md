# Credit-Risk Probability of Default Project
This is a machine learning project using LendingClub loan data from 2007-2018 to predict the probability of the borrower defaulting. It is a portfolio project to develop skills suitable for quantitative finance, actuarial and risk-modelling roles.

## Overview
The goal of this project is to develop a Probability of Default (PD) model. This is a key aspect of credit risk frameworks used by banks to assess risk and ensure they conform to Basel requirements. This project covers the full process, from initial data preparation and exploration, through to model evaluation and stability testing.

## Data
The data used was found on Kaggle - All Lending Club loan data (Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club). There were initially over 2 million rows and 151 columns but this was cut to around 1.3 million rows and 38 columns after cleaning.  
The target variable was 'default', which was binary - 1 if the loan defaulted and 0 if it was fully paid. Alongside this, there were 37 variables covering borrower financial information, credit history and loan characteristics, made up of both numerical and categorical variables. All unresolved loans were excluded.
