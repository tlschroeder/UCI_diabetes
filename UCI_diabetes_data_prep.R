library(dplyr)

##load data and apply recommended transformations
uciData <- read.csv("dataset_diabetes/diabetic_data.csv", na.strings = "?")

factors <- c('admission_type_id', 'discharge_disposition_id','admission_source_id') #uncaught factors
uciData[,factors] <- lapply(uciData[,factors],factor)
uciData[,c('encounter_id', 'patient_nbr')] <- NULL #no patient-level info used

#narrow down response to metric of interest
vals <- c('NO' = 0, '>30' = 0, '<30' = 1)
uciData$readmitted <- vals[as.character(uciData$readmitted)]

#simplify medications
uciData[,23:45] <- unlist(apply(uciData[,23:45],2,function(x)
  {vals <- c('No' = 0,'Steady' = 1,'Up' = 1,'Down' = 1)
  return(vals[as.character(x)])})) 

#check >95% single-value variables
exclude <- unlist(apply(uciData[,],2,function(x){
  return(max(table(x, useNA = 'always'))/length(x) >= .95)
}))
uciData[,'weight'] <- NULL #drop weight for being mostly NA, let xgb/clustering handle the rest
uciUnPivoted <- uciData

#pivoting diagnoses out
allDx <- union(levels(uciData$diag_1), levels(uciData$diag_2)) %>% union(levels(uciData$diag_3))

dxFrame <- data.frame(lapply(allDx, function(x){
  d1 <- (uciData$diag_1 == x)
  d1[is.na(d1)] <- FALSE #remove NAs to avoid wiping out rows
  d2 <- (uciData$diag_2 == x)
  d2[is.na(d2)] <- FALSE
  d3 <- (uciData$diag_3 == x)
  d3[is.na(d3)] <- FALSE
  return(d1 | d2 | d3)
}))
names(dxFrame) <- allDx
dxFrame <- lapply(dxFrame,as.numeric)
uciData[,c('diag_1','diag_2','diag_3')] <- NULL
uciData <- cbind(uciData,dxFrame)

saveRDS(list(uciData, uciUnPivoted, names(dxFrame)),'cleanData.RDS')
