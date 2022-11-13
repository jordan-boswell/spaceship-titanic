# Thuy: '~/Documents/GitHub/spaceship-titanic'
# Jordan: 'C:/Users/jbos1/Desktop/Projects/Kaggle/spaceship-titanic'
# Jordan Laptop: 'C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic'
setwd('C:/Users/User/Documents/Projects/Kaggle/spaceship-titanic')

library(tidymodels)
library(vip)
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
  size = 200
)

set.seed(1)
xg_res <- xg_wf %>%
  tune_grid(resamples = folds,
            grid = xg_grid,
            control = control_grid(save_pred = T))

xg_best <- select_best(xg_res, 'accuracy')
xg_oos_pred <- do.call(rbind, lapply(xg_res$.predictions, function(x){x[x$.config == xg_best$.config, c('.row', '.pred_TRUE')]}))
xg_oos_pred <- xg_oos_pred[order(xg_oos_pred$.row), ]
savePredictions(train$PassengerId, xg_oos_pred$.pred_TRUE, 'xg_oos')

xg_final_wf <- xg_wf %>% 
  finalize_workflow(xg_best)
xg_final_fit <- xg_final_wf %>% fit(train)
xg_pred <-predict(xg_final_fit, test, type = 'prob')$.pred_TRUE
savePredictions(test$PassengerId, xg_pred, "xg_1")


## Random Forest
rf_spec <- rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) %>%
  set_engine('randomForest') %>%
  set_mode('classification')

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rec)

set.seed(1)
rf_grid <-
  grid_latin_hypercube(mtry() %>% range_set(c(1, num_cols)),
                       trees(),
                       min_n(),
                       size = 200)

set.seed(1)
rf_res <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = rf_grid,
            control = control_grid(save_pred = T))

rf_best <- select_best(rf_res, 'accuracy')
rf_oos_pred <- do.call(rbind, lapply(rf_res$.predictions, function(x){x[x$.config == rf_best$.config, c('.row', '.pred_TRUE')]}))
rf_oos_pred <- rf_oos_pred[order(rf_oos_pred$.row), ]
savePredictions(train$PassengerId, rf_oos_pred$.pred_TRUE, 'rf_oos')

rf_final_wf <- rf_wf %>% 
  finalize_workflow(select_best(rf_res, "accuracy"))
rf_final_fit <- rf_final_wf %>% fit(train)
rf_pred <- predict(rf_final_fit, test, type = 'prob')$.pred_TRUE
savePredictions(test$PassengerId, rf_pred, "rf_1")


## Lasso
ls_spec <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = 1) %>%
  set_engine ( "glmnet" )

ls_rec <- rec 

ls_wf <- workflow() %>% add_model(ls_spec) %>% add_recipe(ls_rec)

ls_grid <- tibble(penalty=10^seq(-4, -0.5, length.out = 200))
                                 
set.seed(1)
ls_res <- ls_wf %>%
  tune_grid(resamples = folds,
            grid = ls_grid,
            control = control_grid(save_pred = T))

ls_best <- select_best(ls_res, 'accuracy')
ls_oos_pred <- do.call(rbind, lapply(ls_res$.predictions, function(x){x[x$.config == ls_best$.config, c('.row', '.pred_TRUE')]}))
ls_oos_pred <- ls_oos_pred[order(ls_oos_pred$.row), ]
savePredictions(train$PassengerId, ls_oos_pred$.pred_TRUE, 'ls_oos')

ls_final_wf <- ls_wf %>% 
  finalize_workflow(select_best(ls_res, "accuracy"))
ls_final_fit <- ls_final_wf %>% fit(train)
ls_pred <- predict(ls_final_fit, test, type = 'prob')$.pred_TRUE
savePredictions(test$PassengerId, ls_pred, "lasso_3")


# Meta Model
xg_oos_pred <- read.csv('submissions/xg_oos.csv')
rf_oos_pred <- read.csv('submissions/rf_oos.csv')
ls_oos_pred <- read.csv('submissions/ls_oos.csv')
oos_pred <- data.frame(PredXG = xg_oos_pred$Transported>0.5, PredRF = rf_oos_pred$Transported>0.5, PredLS = ls_oos_pred$Transported>0.5)
test_pred <- data.frame(PredXG = xg_pred, PredRF = rf_pred, PredLS = ls_pred)
meta_train <- cbind(train, oos_pred)
meta_test <- cbind(test, test_pred)

meta_spec <- logistic_reg()
meta_rec <- recipe(Transported ~ PredXG + PredRF + PredLS, data = meta_train)
meta_wf <- workflow() %>% add_model(meta_spec) %>% add_recipe(meta_rec)
meta_final_fit <- meta_wf %>% fit(meta_train)

meta_pred <- (meta_final_fit %>% predict(meta_test, type = 'prob'))$.pred_TRUE

savePredictions(ship_imp_nores[ship_imp_nores$Train == 'FALSE', 'PassengerId'], meta_pred > 0.5, 'meta_2')
