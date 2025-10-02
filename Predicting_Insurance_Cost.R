setwd('/Users/caitlynpratt/Documents/2024 Spring/DTSC 3010 Intro to Data Science')
med <- read.csv('medical_insurance.csv')
getwd()
#1. STATISTICAL SUMMARY
summary(med)


#2. VISUALIZATION PLOT
#A.children
library(ggplot2)


ggplot(data = med) + geom_point(mapping = aes(x = children, y = charges, color = children))



#age

ggplot(data = med) + geom_point(mapping = aes(x = age, y = charges, color = age))

#bmi


ggplot(data = med) + geom_point(mapping = aes(x = bmi, y = charges, color = bmi))


#3. STATISTICAL TEST

#correlation tests
cor.test(med$children, med$charges)
cor.test(med$age, med$charges)
cor.test(med$bmi, med$charges)


#4. MODELS
# model 1: Penalized Linear Regression
head(med)
sample_size <- floor(0.8*nrow(med))
sample_size
train_ind <- sample(seq_len(nrow(med)), size = sample_size)
train <- med[train_ind,]
test <- med[-train_ind,]
train <- train[complete.cases(train),]
test <- test[complete.cases(test),]
library("caret")
fitControl <- trainControl(
  method = 'repeatedcv',
  number = 5,
  repeats = 2)

plrFit <- train(charges~age+bmi+children, data = train,
                method = "penalized",
                trControl=fitControl)
plrFit
prediction_plrFit <- predict(plrFit, newdata = test)
library(caret)
RMSE(prediction_plrFit, test$charges)
cor.test(prediction_plrFit, test$charges)


#B.model 2:random forest 

randomforestfit <- train(charges~age+bmi+children, data = train, method = "ranger", trControl = fitControl)
randomforestfit
prediction_rf <- predict(randomforestfit, newdata = test)
RMSE(prediction_rf, test$charges)
cor.test(prediction_rf, test$charges)


## model 3: L2 Regularized Support Vector Machine with Linear Kernel
svmfit <- train(charges~age+bmi+children, data = train, method = "svmLinear3", trControl = fitControl)
svmfit
prediction_rf <- predict(svmfit, newdata = test)
RMSE(prediction_rf, test$charges)
cor.test(prediction_rf, test$charges)


library(xgboost)

installed.packages()["xgboost", ]

# Prepare data
train_matrix <- model.matrix(charges ~ . -1, data = train)
train_labels <- train$charges
test_matrix <- model.matrix(charges ~ . -1, data = test)

dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)
dtest <- xgb.DMatrix(data = test_matrix)

# Set parameters (you can tune these manually)
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  verbose = 0
)

# Predict and evaluate
xgb_pred <- predict(xgb_model, newdata = dtest)
xgb_rmse <- sqrt(mean((xgb_pred - test$charges)^2))
print(paste("XGBoost RMSE:", round(xgb_rmse, 2)))
cor.test(xgb_pred, test$charges)
RMSE(xgb_pred, test$charges)
