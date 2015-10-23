#install.packages("caret")
#install.packages("pROC")
require(caret)
require(pROC)

ctrl <- trainControl('cv', number=3, summaryFunction=twoClassSummary,
                     verboseIter=TRUE, classProbs=TRUE)
nroundlimit<-100
xgb.grid <- expand.grid(nrounds=2:nroundlimit, max_depth=c(6, 8),
                        eta=c(0.1, 0.35))
#train$target<-as.factor(train$target,labels=c("X0","X1"))

target<-as.factor(ifelse(train$target==0,"X0","X1"))

fit.xgbTree <- train(x=train[setdiff(names(train),'target')], y=target, 
                     method='xgbTree', metric='ROC', trControl=ctrl, 
                     tuneGrid=xgb.grid, subsample=0.5, colsample.bytree=0.8)