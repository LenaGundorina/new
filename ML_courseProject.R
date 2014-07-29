library("caret")
library("kernlab")
library("hydroGOF")
library("Metrics")
library("randomForest")


source('C:/Users/HundorynaA/Downloads/Practical Machine Learning/findColumnsToDrop.R')

setwd("C:/Users/HundorynaA/Downloads")
quizData <- read.csv("pml-testing.csv")
originData <- read.csv("pml-training.csv")
summary(originData)
colSums(is.na(originData))

##Drop columns where more than 80% of data is missing
columnsToDrop <- apply(originData,2,findColumnsToDrop,threshold=80,hundred_percents=dim(originData)[1])
columnsToDrop <- columnsToDrop[columnsToDrop ==T]
filteredData <- originData[,!(names(originData) %in% names(columnsToDrop))]
quizData <- quizData[,!(names(quizData) %in% names(columnsToDrop))]

## Identification and reduction of near zero variance predictors
nearZeroVar <- nearZeroVar(filteredData, saveMetrics=T)
columnNames <- names(filteredData)
appropriateColumnNames <- columnNames[!nearZeroVar$nzv ]
filteredData <- filteredData[,names(filteredData) %in% appropriateColumnNames]
quizData <- quizData[,names(quizData) %in% appropriateColumnNames]

##Data slicing
inTrain <- createDataPartition(y=filteredData$classe,p=0.75,list=F)
training <- filteredData[inTrain,]
testing <-  filteredData[-inTrain,]

## PCA
# drop factors
isFactor <- filteredData[,sapply(filteredData,is.factor)]
training <- training[,!(names(training) %in% names(isFactor)[1:2])]
testing <- testing[,!(names(testing) %in% names(isFactor)[1:2])]
quizData <- quizData[,!(names(quizData) %in% names(isFactor)[1:2])]
pca_training <- preProcess(training[,-57],method="pca",thresh = 0.85) 
trainingPC <- predict(pca_training,training[,-57])
testingPC <- predict(pca_training,testing[,-57])
quizDataPC <- predict(pca_training,quizData)

##Training Prediction
##LR
set.seed(100)
modelFit <- train(training$classe~., method="lm",data=trainingPC)
##Decision Trees
modelFit <- train(training$classe~., method="rpart",data=trainingPC)
##Bagging
modelFit <- train(training$classe~., method="bagEarth",data=trainingPC)
##Random Forest
modelFit <- train(training$classe~., method="rf",data=trainingPC,prox=T)
modelFit <- train(training$classe~., method="parRF",data=trainingPC,prox=T)
modelFit <- randomForest(training$classe~.,data=trainingPC,importance=T)

##Boosting
modelFit <- train(training$classe~., method="gbm",data=trainingPC,verbose=F)
modelFit <- gbm(training$classe~.,trainingPC)

modelFit$finalModel


##Prediction
predictions <- predict(modelFit,testingPC)
RFpredictions <- predict(modelFit,testingPC)
quiz_predictions <- predict(modelFit,quizDataPC,)

##Error rate
table(predictions,testing$classe)
table(RFpredictions,testing$classe)
confusionMatrix(predictions,testing$classe)
confusionMatrix(RFpredictions,testing$classe)
_________________________________________________________________________________

### Functions

findColumnsToDrop <- function(x,threshold,hundred_percents){
  i <- F
  columnsToDrop <- logical()
  if((sum(is.na(x))*100/hundred_percents)>=threshold){
    i <- T
  }
  columnsToDrop <- append(columnsToDrop, i)
  return(columnsToDrop)
}
