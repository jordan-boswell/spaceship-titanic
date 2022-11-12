library(missForest)
library(doParallel)
registerDoParallel(cores=12)

# Thuy: '~/Documents/GitHub/spaceship-titanic'
# Jordan: 'C:/Users/jbos1/Desktop/Projects/Kaggle/spaceship-titanic'
# Jordan Laptop: 'C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic'
setwd('C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic')
source('analysis/0 Functions.R')
ship <- read.csv('data/ship.csv')

ship <- setColTypesForModeling(ship)

imputation_cols <- c('Age', 'CabinSize', 'CryoSleep', 'Deck', 'Destination', 'GID', 'GroupSize', 'HasSpent', 'HomePlanet', 'IID', 'Num', 'RoomService', 'ShoppingMall', 'Side', 'Spa', 'Spending', 'Transported', 'VIP', 'VRDeck')
set.seed(1)
# Note: ntree values of 10,20,30,40,50 took 126,292,493,571,718 seconds, respectively
begin_time <- proc.time()
rf <- missForest(ship[, imputation_cols], maxiter=30, ntree=1000, parallelize="variables")
print(begin_time - proc.time())
ship_imputed_with_response <- rf$ximp

ship_imputed_with_response <- cbind(ship_imputed_with_response, ship[, !(names(ship) %in% names(ship_imputed_with_response))])
ship_imputed_with_response <- updateSpending(ship_imputed_with_response)
ship_imputed_with_response <- updateSpendingOfCryo(ship_imputed_with_response)
ship_imputed_with_response <- updateSpendingOfChildren(ship_imputed_with_response)
ship_imputed_with_response <- updateHomePlanetFromDeck(ship_imputed_with_response)
ship_imputed_with_response <- updateVIPFromHomePlanet(ship_imputed_with_response)
ship_imputed_with_response$Cabin <- paste(ship_imputed_with_response$Deck, as.integer(ship_imputed_with_response$Num), ship_imputed_with_response$Side, sep='/')

write.csv(ship_imputed_with_response, 'data/ship_imputed_with_response.csv', row.names=F)