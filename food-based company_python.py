
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

cc = pd.read_csv(r"C:\Users\UMANG\OneDrive\Desktop\Simple/calories_consumed.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

cc.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = cc.calories, x = np.arange(1, 15, 1))
plt.hist(cc.calories) #histogram
plt.boxplot(cc.calories) #boxplot

plt.bar(height = cc.weight, x = np.arange(1, 15, 1))
plt.hist(cc.weight) #histogram
plt.boxplot(cc.weight) #boxplot

# Scatter plot
plt.scatter(x = cc['weight'], y = cc['calories'], color = 'green') 

# correlation
np.corrcoef(cc.weight, cc.calories) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(cc.weight, cc.calories)[0, 1]
cov_output

cc.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('calories ~ weight', data = cc).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(cc['weight']))

# Regression Line
plt.scatter(cc.weight, cc.calories)
plt.plot(cc.weight, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = cc.calories - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(cc['weight']), y = cc['calories'], color = 'brown')
np.corrcoef(np.log(cc.weight), cc.calories) #correlation

model2 = smf.ols('calories ~ np.log(weight)', data = cc).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(cc['weight']))

# Regression Line
plt.scatter(np.log(cc.weight), cc.calories)
plt.plot(np.log(cc.weight), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = cc.calories - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = cc['weight'], y = np.log(cc['calories']), color = 'orange')
np.corrcoef(cc.weight, np.log(cc.calories)) #correlation

model3 = smf.ols('np.log(calories) ~ weight', data = cc).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(cc['weight']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(cc.weight, np.log(cc.calories))
plt.plot(cc.weight, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = cc.calories - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(calories) ~ weight + I(weight*weight)', data = cc).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(cc))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = cc.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(cc.weight, np.log(cc.calories))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = cc.calories - pred4_at
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

train, test = train_test_split(cc, test_size = 0.2)

finalmodel = smf.ols('np.log(calories) ~ weight + I(weight*weight)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_calories = np.exp(test_pred)
pred_test_calories

# Model Evaluation on Test data
test_res = test.calories - pred_test_calories
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_calories = np.exp(train_pred)
pred_train_calories

# Model Evaluation on train data
train_res = train.calories - pred_train_calories
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

