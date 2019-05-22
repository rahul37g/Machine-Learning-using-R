# https://machinelearningmastery.com/machine-learning-in-r-step-by-step/

# The caret package (short for Classification And REgression Training) is a set of functions 
# that attempt to streamline the process for creating predictive models.
library(caret)

## 1. Load Data
data(iris)
dataset <- iris # rename the dataset

## 2. Create a validation set
# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index, ]

# use the remaning 80% of dta to training and testing the models
dataset <- dataset[validation_index, ]

## 3. Summarize Dataset
dim(dataset) # dimension of dataset
sapply(dataset, class) # list types for each attributes
head(dataset) # peek at the data
levels(dataset$Species) # list the levels of the class

# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# statistical summary
summary(dataset) 

## 4. Visualize dataset
# Unvariate plots (To better understand each attribute)
# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for (i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

# Multivaraite Plots
featurePlot(x=x, y=y, plot = "ellipse") # scatterplot matrix
featurePlot(x=x, y=y, plot = "box") # # box and whisker plots for each attribute

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot = "density", scales=scales)

## 5. Evaluate Some Algorithms
# Run algortihm using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

# Build Models
# a. linear algorithm
set.seed(7)
fit.lda <- train(Species~., data = dataset, method = "lda", metric=metric, trControl=control)

# b. nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data = dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# c. advanced algorithm
# SVM
set.seed(7)
fit.svm <- train(Species~., data = dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data = dataset, method="rf", metric=metric, trControl=control)

## Select best model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda,  cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize best model
print(fit.lda)

## 6. Make Predictions
# estimate skill of LDA on the validation dataset
prediction <- predict(fit.lda, validation)
confusionMatrix(prediction, validation$Species)
