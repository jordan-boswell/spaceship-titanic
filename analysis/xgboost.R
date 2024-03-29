# Thuy: '~/Documents/GitHub/spaceship-titanic'
# Jordan: 'C:/Users/jbos1/Desktop/Projects/Kaggle/spaceship-titanic'
# Jordan Laptop: 'C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic'
setwd('~/Documents/GitHub/spaceship-titanic')

library(tidymodels)
library(xgboost)

library(doParallel)
doParallel::registerDoParallel()
source('analysis/0 Functions.R')

# read in data
ship <- read.csv('data/ship.csv')
ship_imp_res <- read.csv('data/ship_imputed_with_response.csv')
ship_imp_nores <- read.csv('data/ship_imputed_no_response.csv')

# change column types to numeric and factor
ship <- setColTypesForModeling(ship)
ship_imp_res <- setColTypesForModeling(ship_imp_res)
ship_imp_nores <- setColTypesForModeling(ship_imp_nores)

cols <- c('Age', 'CabinSize', 'CryoSleep', 'Deck', 'Destination', 'GID', 'GroupSize', 'HasSpent', 'HomePlanet', 'IID', 'Num', 'RoomService', 'ShoppingMall', 'Side', 'Spa', 'Spending', 'Transported', 'VIP', 'VRDeck')

num_cols <- numDesignMatColsFromDataset(ship[, cols])

split_indices <-
  list(
    analysis = which(ship$Train == 'TRUE'),
    assessment = which(ship$Train == 'FALSE')
  )
splits <- make_splits(split_indices, ship[, cols])
ship_train <- training(splits)
ship_test <- testing(splits)
folds <- vfold_cv(ship_train)

rec <- recipe(Transported ~ ., data = ship_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_unknown()

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

set.seed(1)
xg_grid <- grid_latin_hypercube(
  mtry() %>% range_set(c(1, num_cols)),
  trees(),
  min_n(),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(),
  size = 3
)

set.seed(1)
xg_res <- xg_wf %>%
  tune_grid(resamples = folds,
            grid = xg_grid,
            control = control_grid(save_pred = T))

xg_final_wf <- xg_wf %>%
  finalize_workflow(select_best(xg_res, 'accuracy'))

xg_final_fit <- xg_final_wf %>% fit(ship_train)

predictions <- collect_predictions(xg_final_fit)$Transported
write.csv(
  data.frame(
    PassengerId = ship$PassengerId[ship$Train == 'FALSE'],
    Transported = ifelse(predictions == 'TRUE', 'True', 'False')
  ),
  file = 'submissions/basic_xg.csv',
  quote = F,
  row.names = F
)
