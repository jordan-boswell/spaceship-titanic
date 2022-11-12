#load data
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
ship_imp_nores <- read.csv('data/ship_imputed_no_response.csv')

#format the data
ship_imp_nores <- setColTypesForModeling(ship_imp_nores)

#get the number of the cols of the model matrix
cols <- c('Age', 'CabinSize', 'CryoSleep', 'Deck', 'Destination', 'GID', 'GroupSize', 'HasSpent', 'HomePlanet', 'IID', 'Num', 'RoomService', 'ShoppingMall', 'Side', 'Spa', 'Spending', 'Transported', 'VIP', 'VRDeck')
num_cols <- numDesignMatColsFromDataset(ship[, cols])

# create the recipe
rec <- recipe(Transported ~ ., data = ship_imp_nores[, cols]) %>%
  step_dummy(all_nominal_predictors())

#create the folds
folds <- vfold_cv(ship_imp_nores[, cols])

#base models
## xgboost
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
  size = 30
)

set.seed(1)
xg_res <- xg_wf %>%
  tune_grid(resamples = folds,
            grid = xg_grid,
            control = control_grid(save_pred = T))

## Lasso
ls_spec <- logistic_reg(mode = "classification",
                        penalty = tune(),
                        mixture = 1,
                        set_engine = "glmnet")

#meta model