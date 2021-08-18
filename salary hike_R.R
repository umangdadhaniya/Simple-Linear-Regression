# Load the data
YS <- read.csv(file.choose(), header = T)
View(YS)

# Exploratory data analysis
summary(YS)

install.packages("Hmisc")
library(Hmisc)
describe(YS)


install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(YS$YearsExperience, main = "YearsExperience")
dotplot(YS$Salary, main = "Dot Plot Salary")

?boxplot
boxplot(YS$YearsExperience, col = "dodgerblue4")
boxplot(YS$Salary, col = "red", horizontal = T)

hist(YS$YearsExperience)
hist(YS$Salary)

# Normal QQ plot
qqnorm(YS$YearsExperience)
qqline(YS$YearsExperience)

qqnorm(YS$Salary)
qqline(YS$Salary)

hist(YS$YearsExperience, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(YS$YearsExperience))             # add a density estimate with defaults
lines(density(YS$YearsExperience, adjust = 3), lty = "dotted")   # add another "smoother" density

hist(YS$Salary, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(YS$Salary))             # add a density estimate with defaults
lines(density(YS$Salary, adjust = 3), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(YS$YearsExperience, YS$Salary, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "YearsExperience", 
     ylab = "Salary", pch = 20)  # plot(x,y)



## alternate simple command
plot(YS$YearsExperience, YS$Salary)

attach(YS)

# Correlation Coefficient
cor(YearsExperience, Salary)

# Covariance
cov(YearsExperience, Salary)

# Linear Regression model
reg <- lm(Salary ~ YearsExperience, data = YS) # Y ~ X
?lm
summary(reg)

confint(reg, level = 0.95)
?confint

pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)

View(pred)
?predict

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = YS, aes(YearsExperience,Salary) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = YS, aes(x = YearsExperience, y = Salary)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = YS, aes(x = YearsExperience, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, YS$Salary)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(YearsExperience), Salary)
cor(log(YearsExperience), Salary)

reg_log <- lm(Salary ~ log(YearsExperience), data = YS)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, YS$Salary)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = YS, aes(log(YearsExperience), Salary) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
#ggplot(data = , aes(x = log(Waist), y = AT)) + 
# geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = log(Waist), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(YearsExperience, log(Salary))
cor(YearsExperience, log(Salary))

reg_log1 <- lm(log(Salary) ~ YearsExperience, data = YS)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, YS$Salary)

res_log1 = Salary - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = YS, aes(YearsExperience, log(Salary)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = YS, aes(x = YearsExperience, y = log(Salary))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = YS, aes(x = YearsExperience, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience), data = YS)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, YS$Salary)

res2 = Salary - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = YS, aes(YearsExperience, log(Salary)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
#ggplot(data = fb, aes(x = weight + I(weight*weight), y = log(calories))) + 
#geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = weight + I(weight^2), y = predlog$fit))


# Data Partition

# Random Sampling
n <- nrow(YS)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- YS[train_ind, ]
test <-  YS[-train_ind, ]

# Non-random sampling
train <- YS[1:90, ]
test <- YS[91:109, ]

plot(train$YearsExperience, log(train$Salary))
plot(test$YearsExperience, log(test$Salary))

model <- lm(log(calories) ~ weight + I(weight * weight), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Salary - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Salary - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse
