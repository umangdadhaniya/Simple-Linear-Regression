# Load the data
SG <- read.csv(file.choose(), header = T)
View(SG)

# Exploratory data analysis
summary(SG)

install.packages("Hmisc")
library(Hmisc)
describe(SG)


install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(SG$SAT_Scores, main = "SAT_Scores")
dotplot(SG$GPA, main = "Dot Plot GPA")

?boxplot
boxplot(SG$SAT_Scores, col = "dodgerblue4")
boxplot(SG$GPA, col = "red", horizontal = T)

hist(SG$SAT_Scores)
hist(SG$GPA)

# Normal QQ plot
qqnorm(SG$SAT_Scores)
qqline(SG$SAT_Scores)

qqnorm(SG$GPA)
qqline(SG$GPA)

hist(SG$SAT_Scores, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(SG$SAT_Scores))             # add a density estimate with defaults
lines(density(SG$SAT_Scores, adjust = 3), lty = "dotted")   # add another "smoother" density

hist(SG$GPA, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(SG$GPA))             # add a density estimate with defaults
lines(density(SG$GPA, adjust = 3), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(SG$SAT_Scores, SG$GPA, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "SAT_Scores", 
     ylab = "GPA", pch = 20)  # plot(x,y)



## alternate simple command
plot(SG$SAT_Scores, SG$GPA)

attach(SG)

# Correlation Coefficient
cor(SAT_Scores, GPA)

# Covariance
cov(SAT_Scores, GPA)

# Linear Regression model
reg <- lm(GPA ~ SAT_Scores, data = SG) # Y ~ X
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

ggplot(data = SG, aes(SAT_Scores,GPA) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = SG, aes(x = SAT_Scores, y = GPA)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = SG, aes(x = SAT_Scores, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, SG$GPA)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(SAT_Scores), GPA)
cor(log(SAT_Scores), GPA)

reg_log <- lm(GPA ~ log(SAT_Scores), data = SG)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, SG$GPA)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = SG, aes(log(SAT_Scores), GPA) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
#ggplot(data = , aes(x = log(Waist), y = AT)) + 
# geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = log(Waist), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(SAT_Scores, log(GPA))
cor(SAT_Scores, log(GPA))

reg_log1 <- lm(log(GPA) ~ SAT_Scores, data = SG)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, SG$GPA)

res_log1 = GPA - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = SG, aes(SAT_Scores, log(GPA)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = SG, aes(x = SAT_Scores, y = log(GPA))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = SG, aes(x = SAT_Scores, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores), data = SG)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, SG$GPA)

res2 = GPA - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = SG, aes(SAT_Scores, log(GPA)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
#ggplot(data = fb, aes(x = weight + I(weight*weight), y = log(calories))) + 
#geom_point(color = 'blue') +
#geom_line(color = 'red', data = wc.at, aes(x = weight + I(weight^2), y = predlog$fit))


# Data Partition

# Random Sampling
n <- nrow(SG)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- SG[train_ind, ]
test <-  SG[-train_ind, ]

# Non-random sampling
train <- SG[1:90, ]
test <- SG[91:109, ]

plot(train$SAT_Scores, log(train$GPA))
plot(test$SAT_Scores, log(test$GPA))

model <- lm(log(calories) ~ weight + I(weight * weight), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$GPA - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$GPA - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse
