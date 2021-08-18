# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 00:28:36 2021

@author: UMANG
"""


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

sg = pd.read_csv(r"C:\Users\UMANG\OneDrive\Desktop\Simple/SAT_GPA.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

sg.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = sg.GPA, x = np.arange(1, 201, 1))
plt.hist(sg.GPA) #histogram
plt.boxplot(sg.GPA) #boxplot

plt.bar(height = sg.SAT_Scores, x = np.arange(1, 201, 1))
plt.hist(sg.SAT_Scores) #histogram
plt.boxplot(sg.SAT_Scores) #boxplot

# Scatter plot
plt.scatter(x = sg['SAT_Scores'], y = sg['GPA'], color = 'green') 

# correlation
np.corrcoef(sg.SAT_Scores, sg.GPA) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(sg.SAT_Scores, sg.GPA)[0, 1]
cov_output

sg.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('GPA ~ SAT_Scores', data = sg).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(sg['SAT_Scores']))

# Regression Line
plt.scatter(sg.SAT_Scores, sg.GPA)
plt.plot(sg.SAT_Scores, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = sg.GPA - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(sg['SAT_Scores']), y = sg['GPA'], color = 'brown')
np.corrcoef(np.log(sg.SAT_Scores), sg.GPA) #correlation

model2 = smf.ols('GPA ~ np.log(SAT_Scores)', data = sg).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sg['SAT_Scores']))

# Regression Line
plt.scatter(np.log(sg.SAT_Scores), sg.GPA)
plt.plot(np.log(sg.SAT_Scores), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = sg.GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = sg['SAT_Scores'], y = np.log(sg['GPA']), color = 'orange')
np.corrcoef(sg.SAT_Scores, np.log(sg.GPA)) #correlation

model3 = smf.ols('np.log(GPA) ~ SAT_Scores', data = sg).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(sg['SAT_Scores']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(sg.SAT_Scores, np.log(sg.GPA))
plt.plot(sg.SAT_Scores, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = sg.GPA - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = sg).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(sg))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = sg.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(sg.SAT_Scores, np.log(sg.GPA))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = sg.GPA - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(sg, test_size = 0.2)

finalmodel = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_GPA = np.exp(test_pred)
pred_test_GPA

# Model Evaluation on Test data
test_res = test.GPA - pred_test_GPA
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_GPA = np.exp(train_pred)
pred_train_GPA

# Model Evaluation on train data
train_res = train.GPA - pred_train_GPA
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

