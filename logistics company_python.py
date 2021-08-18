# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:45:49 2021

@author: UMANG
"""


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

ds = pd.read_csv(r"C:\Users\UMANG\OneDrive\Desktop\Simple/delivery_time.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

ds.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = ds.sorting, x = np.arange(1,22, 1))
plt.hist(ds.sorting) #histogram
plt.boxplot(ds.sorting) #boxplot

plt.bar(height = ds.delivery, x = np.arange(1, 22, 1))
plt.hist(ds.delivery) #histogram
plt.boxplot(ds.delivery) #boxplot

# Scatter plot
plt.scatter(x = ds['delivery'], y = ds['sorting'], color = 'green') 

# correlation
np.corrcoef(ds.delivery, ds.sorting) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(ds.delivery, ds.sorting)[0, 1]
cov_output

ds.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('sorting ~ delivery', data = ds).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(ds['delivery']))

# Regression Line
plt.scatter(ds.delivery, ds.sorting)
plt.plot(ds.delivery, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = ds.sorting - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(ds['delivery']), y = ds['sorting'], color = 'brown')
np.corrcoef(np.log(ds.delivery), ds.sorting) #correlation

model2 = smf.ols('sorting ~ np.log(delivery)', data = ds).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(ds['delivery']))

# Regression Line
plt.scatter(np.log(ds.delivery), ds.sorting)
plt.plot(np.log(ds.delivery), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = ds.sorting - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = ds['delivery'], y = np.log(ds['sorting']), color = 'orange')
np.corrcoef(ds.delivery, np.log(ds.sorting)) #correlation

model3 = smf.ols('np.log(sorting) ~ delivery', data = ds).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(ds['delivery']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(ds.delivery, np.log(ds.sorting))
plt.plot(ds.delivery, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = ds.sorting - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(sorting) ~ delivery + I(delivery*delivery)', data = ds).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(ds))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = ds.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(ds.delivery, np.log(ds.sorting))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = ds.sorting - pred4_at
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

train, test = train_test_split(ds, test_size = 0.2)

finalmodel = smf.ols('np.log(sorting) ~ delivery + I(delivery*delivery)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_sorting = np.exp(test_pred)
pred_test_sorting

# Model Evaluation on Test data
test_res = test.sorting - pred_test_sorting
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_sorting = np.exp(train_pred)
pred_train_sorting

# Model Evaluation on train data
train_res = train.sorting - pred_train_sorting
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

