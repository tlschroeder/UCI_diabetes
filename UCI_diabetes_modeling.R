library(wordVectors)
library(xgboost)
library(caret)
library(dplyr)
library(ROCR)
library(foreach)
library(doParallel)
library(readr)

###clustering (on train data only)
data <- readRDS('cleandata.RDS')
dxNames <- data[[3]]
uciUnclustered <- data[[1]] #for use in baseline model
uciUnPivoted <- data[[2]] #for clustering

#train/test splitting
train_ind <- sample(seq_len(nrow(uciUnclustered)), size = nrow(uciUnclustered)*.8)
unclusteredTrain <- uciUnclustered[train_ind,]
unclusteredTest <- uciUnclustered[-train_ind,]

unPivotedTrain <- uciUnPivoted[train_ind,]
unPivotedTest <- uciUnPivoted[-train_ind,]

##learn clusters based on train
unclusteredDxTrain <- unPivotedTrain[,c('diag_1','diag_2','diag_3')]
unclusteredDxTest <- unPivotedTest[,c('diag_1','diag_2','diag_3')]

#use word2vec to get 200-length embeddings
unclusteredDxTrain %>% write_delim('unclusteredDxTrain.txt',delim = " ", na="", col_names = FALSE)
model <- train_word2vec(train_file = 'unclusteredDxTrain.txt',
                       output_file = "wordVecs.bin",
                       vectors = 200,
                       threads = 8,
                       window  = 5,
                       iter    = 10, 
                       force   = TRUE)

#use k-means to reduce to 25 groups
km.out=kmeans(as.data.frame(model@.Data)[-1,], 
              25, 
              nstart=25, 
              iter.max = 100)

clusters <- data.frame(code = rownames(model)[-1], cluster = km.out$cluster)

#apply cluster identities to full data where each col = count of dx in that cluster
clusters   <- clusters %>% mutate(cluster = paste("Clstr_", cluster, sep="")) 

#train
suppressWarnings({
  clusteredDxTrain <- left_join(unclusteredDxTrain, clusters, by = c('diag_1' = 'code')) %>%
    left_join(clusters, by = c('diag_2' = 'code')) %>%
    left_join(clusters, by = c('diag_3' = 'code')) %>%
    lapply(factor) %>% data.frame
})

clusteredDxTrain[,c('diag_1','diag_2','diag_3')] <- NULL
names(clusteredDxTrain) <- c('dx1','dx2','dx3')

clusterNames <- union(levels(clusteredDxTrain$dx1), levels(clusteredDxTrain$dx2)) %>% union(levels(clusteredDxTrain$dx3))
fullClustersTrain <- data.frame(lapply(clusterNames, function(x){
  x <- rowSums(apply(clusteredDxTrain,2,function(y){return(as.numeric(y == x))}))
}))
names(fullClustersTrain) <- lapply(1:25,function(x){paste('cluster_',x)})
clusteredTrain <- cbind(unPivotedTrain, fullClustersTrain)
clusteredTrain[,c('diag_1','diag_2','diag_3')] <- NULL

#repeated for test
suppressWarnings({
  clusteredDxTest <- left_join(unclusteredDxTest, clusters, by = c('diag_1' = 'code')) %>%
    left_join(clusters, by = c('diag_2' = 'code')) %>%
    left_join(clusters, by = c('diag_3' = 'code')) %>%
    lapply(factor) %>% data.frame
})

clusteredDxTest[,c('diag_1','diag_2','diag_3')] <- NULL
names(clusteredDxTest) <- c('dx1','dx2','dx3')

clusterNames <- union(levels(clusteredDxTest$dx1), levels(clusteredDxTest$dx2)) %>% union(levels(clusteredDxTest$dx3))
fullClustersTest <- data.frame(lapply(clusterNames, function(x){
  x <- rowSums(apply(clusteredDxTest,2,function(y){return(as.numeric(y == x))}))
}))
names(fullClustersTest) <- lapply(1:25,function(x){paste('cluster_',x)})
clusteredTest <- cbind(unPivotedTest, fullClustersTest)
clusteredTest[,c('diag_1','diag_2','diag_3')] <- NULL


###xgboost modeling
##one-hot encode categorical variables
one_hot <- function(dat){
  dummy <- dummyVars(" ~ .", data = dat, fullRank = TRUE)
  return(data.frame(predict(dummy, newdata = dat)))
}
unclusteredTrain <- one_hot(unclusteredTrain)
saveRDS(unclusteredTrain,'unclusteredTrain.RDS')
unclusteredTest <- one_hot(unclusteredTest)
saveRDS(unclusteredTest,'unclusteredTest.RDS')
clusteredTrain <- one_hot(clusteredTrain)
saveRDS(clusteredTrain,'clusteredTrain.RDS')
clusteredTest <- one_hot(clusteredTest)
saveRDS(clusteredTest,'clusteredTest.RDS')
saveRDS(clusters,'clusters.RDS')

##grid search function
xgbSearch <- function(dat,name){
  xgbGrid <- expand.grid(nrounds   = c(10, 50, 100),
                         min_child_weight = c(1,3,5),
                         eta       = c(0.01, 0.08, 0.15),
                         gamma = c(.01,.1,1)
                         )
  validate <- sample(seq_len(nrow(dat)), size = floor(.75*nrow(dat)))                      
  saveRDS(list(dat[,!names(dat) %in% 'readmitted'],dat$readmitted),'modelData.RDS')
  grid_solution <- foreach(i = 1:nrow(xgbGrid), .combine  = rbind, 
                           .export   = c('xgbGrid','validate'),
                           .packages = c('xgboost', 'dplyr', 'ROCR'))  %dopar% { 
                             row <- xgbGrid[i, ]
                             workerdat <- readRDS('modelData.RDS')
                             x <- workerdat[[1]]
                             xtrain <- x[-validate,]
                             y <- workerdat[[2]]
                             ytrain <- y[-validate]
                             #VThis should really be xgb.cv; time and computational restrictions necessitated shortcuts
                             model <- xgboost(data      = xgb.DMatrix(as.matrix(xtrain), label = data.matrix(ytrain)),
                                           nthread   = 8,  
                                           metrics   = list('rmse','auc'), 
                                           verbose   = FALSE,
                                           nrounds   = row$nrounds, 
                                           eta       = row$eta,
                                           gamma     = row$gamma,
                                           min_child_weight = row$min_child_weight,
                                           objective = 'binary:logistic')
                             
                             #obtain AUC
                             xval <- x[validate,]
                             yval <- y[validate]
                             pred <- predict(model, newdata = as.matrix(xval))
                             auc <- prediction(pred,as.factor(yval))
                             auc <- performance(auc,'auc')
                             auc <- as.numeric(auc@y.values)
                             row$auc <- auc
                             return(row)
                           }
  write.csv(grid_solution,paste(gsub('\\:','_',Sys.time()),name))
  return(grid_solution)
}
#limited cores due to memory constraints
registerDoParallel(cores=3) 
clusteredGrid <- xgbSearch(clusteredTrain,'clusteredSolution.csv')
Sys.sleep(600)
registerDoParallel(cores = 2)
unclusteredGrid <- xgbSearch(unclusteredTrain, 'unclusteredSolution.csv') 

##secondary grid search to tune overfitting parameters with cross validation
overfitSearch <- function(dat,tuned,name){
  xgbGrid <- expand.grid(max_depth = c(3, 6, 9),
                         lambda = c(1e-5, .1, 10),
                         alpha = c(1e-5, .1, 10)
  )
  
  saveRDS(list(dat[,!names(dat) %in% 'readmitted'],dat$readmitted),'modelData.RDS') #gets around some Windows memory issues
  grid_solution <- foreach(i = 1:nrow(xgbGrid), .combine  = rbind, 
                           .export   = c('xgbGrid','tuned'),
                           .packages = c('xgboost', 'dplyr', 'ROCR'))  %dopar% { 
                             row <- xgbGrid[i, ]
                             workerdat <- readRDS('modelData.RDS')
                             xtrain <- workerdat[[1]]
                             ytrain <- workerdat[[2]]
                             model <- xgb.cv(data      = xgb.DMatrix(as.matrix(xtrain), label = data.matrix(ytrain)),
                                              nthread   = 8,  
                                              nfold     = 5,
                                              metrics   = list('rmse','auc'), 
                                              verbose   = FALSE,
                                              max_depth = row$max_depth, 
                                              nrounds   = tuned$nrounds, 
                                              eta       = tuned$eta,
                                              gamma     = tuned$gamma,
                                              lambda    = row$lambda,
                                              alpha     = row$alpha,
                                              min_child_weight = tuned$min_child_weight,
                                              objective = 'binary:logistic')
                             
                             #obtain AUC
                             auc <- model$evaluation_log
                             auc <- auc[nrow(auc),'test_auc_mean']
                             row <- cbind(row,auc)
                             names(row) <- c('max_depth','lambda','alpha','auc')
                             return(row)
                           }
  write.csv(grid_solution,paste(gsub('\\:','_',Sys.time()),name))
  return(grid_solution)
}

unclusteredGrid <- read.csv('2017-04-17 03_35_48 unclusteredSolution.csv') #might change these back if the new ones dont perform
unclusteredRow <- unclusteredGrid[unclusteredGrid$auc == max(unclusteredGrid$auc),]
clusteredGrid <- read.csv('2017-04-17 03_11_00 clusteredSolution.csv')
clusteredRow <- clusteredGrid[unclusteredGrid$auc == max(unclusteredGrid$auc),]

#limited cores due to memory constraints
registerDoParallel(cores=3)
clusteredFinalGrid <- overfitSearch(clusteredTrain,clusteredRow,'clusteredFinal.csv')
Sys.sleep(600)
registerDoParallel(cores=2)
unclusteredFinalGrid <- overfitSearch(unclusteredTrain,unclusteredRow,'unclusteredFinal.csv')