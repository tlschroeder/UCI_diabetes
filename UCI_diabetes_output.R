library(xgboost)
library(ggplot2)
library(ggthemes)
library(scales)

###final models and results
unclusteredTrain <- readRDS('unclusteredTrain.RDS')
unclusteredTest <- readRDS('unclusteredTest.RDS')
clusteredTrain <- readRDS('clusteredTrain.RDS')
clusteredTest <- readRDS('clusteredTest.RDS')
clusters <- readRDS('clusters.RDS')

#get test predictions
testModel <- function(grid, finalGrid,train,test,name){
  row1 <- grid[grid$auc == max(grid$auc),]
  row2 <- finalGrid[finalGrid$auc == max(finalGrid$auc),]
  
  row1[,c('X','auc')] <- NULL
  row2[,c('X','auc')] <- NULL
  finalRow <- cbind(row1,row2,name)
  
  xTrain <- train[,!names(train) %in% 'readmitted']
  yTrain <- train[,'readmitted']
  mod <- xgboost(data      = xgb.DMatrix(as.matrix(xTrain), label = data.matrix(yTrain)),
                 nthread   = 8,  
                 metrics   = list('rmse','auc'), 
                 verbose   = FALSE,
                 max_depth = finalRow$max_depth, 
                 nrounds   = finalRow$nrounds, 
                 eta       = finalRow$eta,
                 gamma     = finalRow$gamma,
                 lambda = finalRow$lambda,
                 alpha = finalRow$alpha,
                 min_child_weight = finalRow$min_child_weight,
                 objective = 'binary:logistic')
  xTest <- test[,!names(test) %in% 'readmitted']
  yTest <- test[,'readmitted']
  pred <- predict(mod, newdata = as.matrix(xTest))
  
  #get ROC values
  fullPrediction <- prediction(pred, as.matrix(yTest))
  compare.perf <- performance(fullPrediction, "tpr", "fpr")
  auc <- performance(fullPrediction, measure="auc")@y.values[[1]]
  roc.vals <- data.frame(cbind(compare.perf@x.values[[1]], compare.perf@y.values[[1]]))
  colnames(roc.vals) = c("fp", "tp")
  roc.vals$model <- name
  return(list(roc.vals,auc,mod))
  
}
clusteredOutput <- testModel(read.csv('2017-04-17 03_11_00 clusteredSolution.csv'),
                             read.csv('2017-04-17 04_09_41 clusteredFinal.csv'),
                             clusteredTrain,
                             clusteredTest,
                             'clustered')
roc.vals.clustered <- clusteredOutput[[1]]
clusteredAuc <- clusteredOutput[[2]]
clusteredMod <- clusteredOutput[[3]]
unclusteredOutput <- testModel(read.csv('2017-04-17 03_35_48 unclusteredSolution.csv'),
                               read.csv('2017-04-17 07_04_58 unclusteredFinal.csv'),
                               unclusteredTrain,
                               unclusteredTest,
                               'clustered')
roc.vals.unclustered <- unclusteredOutput[[1]]
unclusteredAuc <- unclusteredOutput[[2]]
unclusteredMod <- unclusteredOutput[[3]]


roc.vals <- rbind(roc.vals.clustered,roc.vals.unclustered)

#plot ROC curve
ggplot(roc.vals, aes(x=fp, y=tp, color = 'model')) + 
  labs(x='false positive rate', y='true positive rate') +
  scale_x_continuous(labels = percent, limits = c(0,1)) + 
  scale_y_continuous(labels = percent, limits = c(0,1)) + 
  geom_abline(aes(intercept=0, slope=1)) + coord_equal() +
  ggtitle(paste("ROC Curve AUC: ", round(max(clusteredAuc,unclusteredAuc),digits=6), sep="")) +
  geom_line(size=1.2, colour="#3498db") 

##write-outs for visualization
importance <- xgb.importance(feature_names = names(clusteredTrain[,!names(clusteredTrain) %in% 'readmitted']),model = clusteredMod)

#names cleanup
importance$Feature <- as.character(importance$Feature)
importance$Feature <- gsub('X\\.','',importance$Feature)
importance$Feature <- gsub('_\\.','_',importance$Feature)
importance$Feature <- gsub('\\.$','',importance$Feature)

#categorization
importance$category <- 'uncategorized'
importance[grep('cluster|number_diag|max_glu|A1C',importance$Feature),'category'] <- 'Diagnosis/Diabetes'
importance[grep('age|gender|race',importance$Feature),'category'] <- 'Demographics'
importance[grep('inpatient|outpatient|emergency',importance$Feature),'category'] <- 'Past Visits'
importance[grep('discharge|admission|payer|time',importance$Feature),'category'] <- 'Administrative'
importance[grep('num_l|num_p|medical_s',importance$Feature),'category'] <- 'Inpatient Treatment'
importance[importance$category == 'uncategorized','category'] <- 'Medication'
write.csv(importance, 'feature importance final.csv')

print(clusters[clusters$cluster == 'Clstr_7','code'])
print(clusters[clusters$cluster == 'Clstr_6','code'])
print(clusters[clusters$cluster == 'Clstr_16','code'])
