setwd('C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic')

library(tidymodels)
library(xgboost)
library(vip)
library(doParallel)
doParallel::registerDoParallel()
source('analysis/0 Shared Functions.R')

# read in data
ship <- read.csv('data/ship.csv')
ship_imp_res <- read.csv('data/ship_imputed_with_response.csv')
ship_imp_nores <- read.csv('data/ship_imputed_no_response.csv')

# change column types to numeric and factor
ship <- setColTypesForModeling(ship)
ship_imp_res <- setColTypesForModeling(ship_imp_res)
ship_imp_nores <- setColTypesForModeling(ship_imp_nores)

cols <- c('Side', 'Deck', 'Num', 'GID', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'IID', 'Spending', 'HasSpent', 'GroupSize', 'GroupNumTransported', 'GroupTransportedPct', 'CabinSize', 'CabinNumTransported', 'CabinTransportedPct', 'SideNeighbors', 'SideNeighborsTransported', 'SideNeighborsTransportedPct', 'BackNeighbors', 'BackNeighborsTransported', 'BackNeighborsTransportedPct', 'DiagFrontNeighbors', 'DiagFrontNeighborsTransported', 'DiagFrontNeighborsTransportedPct', 'DiagBackNeighbors', 'DiagBackNeighborsTransported', 'DiagBackNeighborsTransportedPct')
num_cols <- numDesignMatColsFromDataset(ship)

set.seed(1)

split_indices <- list(analysis = which(ship$Train == 'TRUE'), assessment = which(ship$Train == 'FALSE'))
splits <- make_splits(split_indices, ship[, cols])
ship_train <- training(splits)
ship_test <- testing(splits)
folds <- vfoldcv(ship_train)

rec <- recipe(Transported ~ ., data = ship_train) %>%
  step_dummy(all_nominal_predictors())

xg_spec <- boost_tree(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

xg_wf <- workflow() %>%
  add_model(xg_spec) %>%
  add_recipe(rec)

xg_grid <- grid_latin_hypercube(
  mtry() %>% range_set(c(1, num_cols)),
  trees(),
  min_n(),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(),
  size = 25
)

xg_res <- xg_wf %>%
  tune_grid(
    resamples = folds,
    grid = xg_grid
  )

xg_final_wf <- xg_wf %>%
  finalize_workflow(select_best(xg_res, 'rmse'))

xg_final_fit <- xg_final_wf %>% last_fit(splits)

