# Thuy: '~/Documents/GitHub/spaceship-titanic'
# Jordan: 'C:/Users/jbos1/Desktop/Projects/Kaggle/spaceship-titanic'
# Jordan Laptop: 'C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic'
setwd('C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic')

library(tidymodels)
library(xgboost)
library(randomForest)
library(glmnet)
library(doParallel)
doParallel::registerDoParallel()
source('analysis/0 Functions.R')

# Load the Data
ship_imp_nores <- read.csv('data/ship_imputed_no_response.csv')

# Format the Data
ship_imp_nores <- setColTypesForModeling(ship_imp_nores)
train <- ship_imp_nores[ship_imp_nores$Train == 'TRUE',]
test <- ship_imp_nores[ship_imp_nores$Train == 'FALSE',]

# Get the Columns and Calculate the Number of Columns in the Model Matrix
cols <-
  c(
    'Age',
    'CabinSize',
    'CryoSleep',
    'Deck',
    'Destination',
    'GID',
    'GroupSize',
    'HasSpent',
    'HomePlanet',
    'IID',
    'Num',
    'RoomService',
    'ShoppingMall',
    'Side',
    'Spa',
    'Spending',
    'Transported',
    'VIP',
    'VRDeck'
  )
num_cols <- numDesignMatColsFromDataset(train[, cols])

# Create the Recipe
rec <- recipe(Transported ~ ., data = train[, cols]) %>%
  step_dummy(all_nominal_predictors())

# Create the Folds
set.seed(1)
folds <- vfold_cv(train[, cols])

# Base Models

## XGBoost
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
  size = 2
)

set.seed(1)
xg_res <- xg_wf %>%
  tune_grid(resamples = folds,
            grid = xg_grid,
            control = control_grid(save_pred = T))

xg_final_wf <- xg_wf %>% 
  finalize_workflow(select_best(xg_res, "accuracy"))
xg_final_fit <- xg_final_wf %>% fit(train)
xg_pred <-predict(xg_final_fit, test)
savePredictions(test$PassengerId, xg_pred, "xg_1")


## Random Forest
rf_spec <- rand_forest(mtry = tune(),
                       trees = 2000,
                       min_n = tune()) %>%
  set_engine('randomForest') %>%
  set_mode('classification')

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rec)

set.seed(1)
rf_grid <-
  grid_latin_hypercube(mtry() %>% range_set(c(1, num_cols)),
                       min_n(),
                       size = 2)

set.seed(1)
rf_res <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = rf_grid,
            control = control_grid(save_pred = T))

rf_final_wf <- rf_wf %>% 
  finalize_workflow(select_best(rf_res, "accuracy"))
rf_final_fit <- rf_final_wf %>% fit(train)
rf_pred <- predict(rf_final_fit, test)
savePredictions(test$PassengerId, rf_pred, "rf_1")


## Lasso
ls_spec <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = 1) %>%
  set_engine ( "glmnet" )

ls_rec <- rec 

ls_wf <- workflow() %>% add_model(ls_spec) %>% add_recipe(ls_rec)

ls_grid <-   tibble(penalty=10^seq(-4, -1, length.out = 30))
                                 
set.seed(1)
ls_res <- ls_wf %>%
  tune_grid(resamples = folds,
            grid = ls_grid,
            control = control_grid(save_pred = T))

#Non-stacking Lasso
ls_final_wf <- ls_wf %>% 
  finalize_workflow(select_best(ls_res, "accuracy"))
ls_final_fit <- ls_final_wf %>% fit(train)
ls_pred <- predict(ls_final_fit, test)
savePredictions(test$PassengerId, ls_pred, "lasso_2")
# Meta Model
