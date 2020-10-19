# Regression-ML-Model
Regression model for predicting  Ford cars prices

I have made a supervised machine learning algorithm to predict the price, taking into account top 5 features most correlated with it, using a correlation matrix. Then, I have performed the data cleaning task: fill Na values with median. Categorical data was turned into numerical  by mapping integers each. The manufacturing year had also missing values and I developed another model to fill it in, where the DecisionTreeClassifier scored the best. After using RobustScaler on data, I have plotted the feature heatmap and selected the 5 most correlated estimators with the target variable. Out of models set to evaluate, LinearRegression, Ridge, Lasso, SVR, MLPRegressor, DecisionTreeRegressor performed at the highest, 0.98.
