# Machine-Learning-AA
1. Objective

The objective of this project is to deliver a model which can predict used car prices. Users can interact through the dashboard to input the information of their cars, and the predicted car price will be shown at the end. The potential users of this project can be individuals who wants to know their used car prices, or car trading companies who want to make better pricing strategies. This project is measured at success to deliver a fairly pratical used car price. 


2. Data

Data source in this project is from developer's last project as AA_Datapipeline, which is originally collected from the used car trading company's website 'AA.com'. The data cleaning process includes dealing with missing values and feature engineering. Certain features are reclassified/removed based on domain expertise. Data has been split into train, test, val; standardised without data leakage; random seed is applied for repeatability.


3. Modelling

Different supervised regression models are evaluated includs linear/lasso/ridge/SVM/KNN/decision tree models. The baseline is linear regression model. R2 score is used to evaluate the output, hyperparameter tuning is also used for best result. Best fit model is logged and saved.


4. Presentation

Contains non-technical and technical sections to demostrate the project.
