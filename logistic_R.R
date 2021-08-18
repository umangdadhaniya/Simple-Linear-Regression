# Load the data
ds <- read.csv(file.choose(), header = T)
View(ds)

# Exploratory data analysis
summary(ds)

install.packages("Hmisc")
library(Hmisc)
describe(ds)


install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(ds$delivery, main = "delivery")
dotplot(ds$sorting, main = "Dot Plot sorting")

?boxplot
boxplot(ds$delivery, col = "dodgerblue4")
boxplot(ds$sorting, col = "red", horizontal = T)

hist(ds$delivery)
hist(ds$sorting)

# Normal QQ plot
qqnorm(ds$delivery)
qqline(ds$delivery)

qqnorm(ds$sorting)
qqline(ds$sorting)

hist(ds$delivery, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(ds$delivery))             # add a density estimate with defaults
lines(density(ds$delivery, adjust = 3), lty = "dotted")   # add another "smoother" density

hist(ds$sorting, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(ds$sorting))             # add a density estimate with defaults
lines(density(ds$sorting, adjust = 3), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(ds$delivery, ds$sorting, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "delivery", 
     ylab = "sorting", pch = 20)  # plot(x,y)



## alternate simple command
plot(ds$delivery, ds$sorting)

attach(ds)

# Correlation Coefficient
cor(delivery, sorting)

# Covariance
cov(delivery, sorting)

# Linear Regression model
reg <- lm(sorting ~ delivery, data = ds) # Y ~ X
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

ggplot(data = ds, aes(delivery,sorting) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = ds, aes(x = delivery, y = sorting)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = ds, aes(x = delivery, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, ds$sorting)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(delivery), sorting)
cor(log(delivery), sorting)

reg_log <- lm(sorting ~ log(delivery), data = ds)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, ds$sorting)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = ds, aes(log(delivery), sorting) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
#ggplot(data = , aes(x = log(Waist), y = AT)) + 
# geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = log(Waist), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(delivery, log(sorting))
cor(delivery, log(sorting))

reg_log1 <- lm(log(sorting) ~ delivery, data = ds)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, ds$sorting)

res_log1 = sorting - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = ds, aes(delivery, log(sorting)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = ds, aes(x = delivery, y = log(sorting))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = ds, aes(x = delivery, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(sorting) ~ delivery + I(delivery*delivery), data = ds)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, ds$sorting)

res2 = sorting - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = ds, aes(delivery, log(sorting)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
#ggplot(data = fb, aes(x = weight + I(weight*weight), y = log(calories))) + 
#geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = weight + I(weight^2), y = predlog$fit))


# Data Partition

# Random Sampling
n <- nrow(ds)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- ds[train_ind, ]
test <-  ds[-train_ind, ]

# Non-random sampling
train <- ds[1:90, ]
test <- ds[91:109, ]

plot(train$delivery, log(train$sorting))
plot(test$delivery, log(test$sorting))

model <- lm(log(calories) ~ weight + I(weight * weight), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$sorting - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$sorting - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse
