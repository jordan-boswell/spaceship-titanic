library(missForest)
library(doParallel)
registerDoParallel(cores=12)

# Thuy: '~/Documents/GitHub/spaceship-titanic/data'
# Jordan: 'C:/Users/jbos1/Desktop/Projects/Kaggle/spaceship-titanic/data'
setwd('C:/Users/jbos1/Desktop/Projects/Kaggle/spaceship-titanic/data')
ship <- read.csv('ship.csv')

for (name in names(ship)) {
    col_type <- typeof(ship[, name])
    if (col_type == 'character')
        ship[, name] <- as.factor(ship[, name])
    else if (col_type == 'logical')
        ship[, name] <- as.factor(ship[, name])
    else if (col_type == 'integer')
        ship[, name] <- as.numeric(ship[, name])
}

imputation_cols <- names(ship)[!(names(ship) %in% c('Train', 'PassengerId', 'Cabin', 'Name', 'LastName', 'GroupCabinMode'))]
# Note: ntree values of 10,20,30,40,50 took 126,292,493,571,718 seconds, respectively
begin_time <- proc.time()
rf <- missForest(ship[, imputation_cols], maxiter=30, ntree=1000, parallelize="variables")
print(begin_time - proc.time())
ship_imputed <- rf$ximp

write.csv(ship_imputed, 'ship_imputed.csv', row.names=F)