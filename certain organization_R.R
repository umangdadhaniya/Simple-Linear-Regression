# Load the data
sc <- read.csv(file.choose(), header = T)
View(sc)

# Exploratory data analysis
summary(sc)

install.packages("Hmisc")
library(Hmisc)
describe(sc)


install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(sc$Salary_hike, main = "Salary_hike")
dotplot(sc$Churn_out_rate, main = "Dot Plot Churn_out_rate")

?boxplot
boxplot(sc$Salary_hike, col = "dodgerblue4")
boxplot(sc$Churn_out_rate, col = "red", horizontal = T)

hist(sc$Salary_hike)
hist(sc$Churn_out_rate)

# Normal QQ plot
qqnorm(sc$Salary_hike)
qqline(sc$Salary_hike)

qqnorm(sc$Churn_out_rate)
qqline(sc$Churn_out_rate)

hist(sc$Salary_hike, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(sc$Salary_hike))             # add a density estimate with defaults
lines(density(sc$Salary_hike, adjust = 3), lty = "dotted")   # add another "smoother" density

hist(sc$Churn_out_rate, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(sc$Churn_out_rate))             # add a density estimate with defaults
lines(density(sc$Churn_out_rate, adjust = 3), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(sc$Salary_hike, sc$Churn_out_rate, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Salary_hike", 
     ylab = "Churn_out_rate", pch = 20)  # plot(x,y)



## alternate simple command
plot(sc$Salary_hike, sc$Churn_out_rate)

attach(sc)

# Correlation Coefficient
cor(Salary_hike, Churn_out_rate)

# Covariance
cov(Salary_hike, Churn_out_rate)

# Linear Regression model
reg <- lm(Churn_out_rate ~ Salary_hike, data = sc) # Y ~ X
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

ggplot(data = sc, aes(Salary_hike,Churn_out_rate) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = sc, aes(x = Salary_hike, y = Churn_out_rate)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = sc, aes(x = Salary_hike, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, sc$Churn_out_rate)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(Salary_hike), Churn_out_rate)
cor(log(Salary_hike), Churn_out_rate)

reg_log <- lm(Churn_out_rate ~ log(Salary_hike), data = sc)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, sc$Churn_out_rate)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = sc, aes(log(Salary_hike), Churn_out_rate) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
#ggplot(data = , aes(x = log(Waist), y = AT)) + 
# geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = log(Waist), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(Salary_hike, log(Churn_out_rate))
cor(Salary_hike, log(Churn_out_rate))

reg_log1 <- lm(log(Churn_out_rate) ~ Salary_hike, data = sc)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, sc$Churn_out_rate)

res_log1 = Churn_out_rate - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = sc, aes(Salary_hike, log(Churn_out_rate)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = sc, aes(x = Salary_hike, y = log(Churn_out_rate))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = sc, aes(x = Salary_hike, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike), data = sc)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, sc$Churn_out_rate)

res2 = Churn_out_rate - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = sc, aes(Salary_hike, log(Churn_out_rate)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
#ggplot(data = fb, aes(x = weight + I(weight*weight), y = log(calories))) + 
#geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = weight + I(weight^2), y = predlog$fit))


# Data Partition

# Random Sampling
n <- nrow(sc)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- sc[train_ind, ]
test <-  sc[-train_ind, ]

# Non-random sampling
train <- sc[1:90, ]
test <- sc[91:109, ]

plot(train$Salary_hike, log(train$Churn_out_rate))
plot(test$Salary_hike, log(test$Churn_out_rate))

model <- lm(log(calories) ~ weight + I(weight * weight), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Churn_out_rate - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Churn_out_rate - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse

