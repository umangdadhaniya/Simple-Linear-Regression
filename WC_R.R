# Load the data
fb <- read.csv(file.choose(), header = T)
View(fb)

# Exploratory data analysis
summary(fb)

install.packages("Hmisc")
library(Hmisc)
describe(fb)


install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(fb$weight, main = "weight")
dotplot(fb$calories, main = "Dot Plot calories")

?boxplot
boxplot(fb$weight, col = "dodgerblue4")
boxplot(fb$calories, col = "red", horizontal = T)

hist(fb$weight)
hist(fb$calories)

# Normal QQ plot
qqnorm(fb$weight)
qqline(fb$weight)

qqnorm(fb$calories)
qqline(fb$calories)

hist(fb$weight, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(fb$weight))             # add a density estimate with defaults
lines(density(fb$weight, adjust = 2), lty = "dotted")   # add another "smoother" density

hist(fb$calories, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(fb$calories))             # add a density estimate with defaults
lines(density(fb$calories, adjust = 3), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(fb$weight, fb$calories, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "weight", 
     ylab = "calories", pch = 20)  # plot(x,y)



## alternate simple command
plot(fb$weight, fb$calories)

attach(fb)

# Correlation Coefficient
cor(weight, calories)

# Covariance
cov(weight, calories)

# Linear Regression model
reg <- lm(calories ~ weight, data = fb) # Y ~ X
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

ggplot(data = fb, aes(weight,calories) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = fb, aes(x = weight, y = calories)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = fb, aes(x = weight, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, fb$calories)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(weight), calories)
cor(log(weight), calories)

reg_log <- lm(calories ~ log(weight), data = fb)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, fb$calories)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = fb, aes(log(weight), calories) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
#ggplot(data = , aes(x = log(Waist), y = AT)) + 
 # geom_point(color = 'blue') +
  #geom_line(color = 'red', data = wc.at, aes(x = log(Waist), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(weight, log(calories))
cor(weight, log(calories))

reg_log1 <- lm(log(calories) ~ weight, data = fb)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, fb$calories)

res_log1 = calories - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = fb, aes(weight, log(calories)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = fb, aes(x = weight, y = log(calories))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = fb, aes(x = weight, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(calories) ~ weight + I(weight*weight), data = fb)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, fb$calories)

res2 = calories - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = fb, aes(weight, log(calories)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
#ggplot(data = fb, aes(x = weight + I(weight*weight), y = log(calories))) + 
  #geom_point(color = 'blue') +
  #geom_line(color = 'red', data = wc.at, aes(x = weight + I(weight^2), y = predlog$fit))


# Data Partition

# Random Sampling
n <- nrow(fb)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- fb[train_ind, ]
test <-  fb[-train_ind, ]

# Non-random sampling
train <- fb[1:90, ]
test <- fb[91:109, ]

plot(train$weight, log(train$calories))
plot(test$weight, log(test$calories))

model <- lm(log(calories) ~ weight + I(weight * weight), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$calories - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$calories - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse
