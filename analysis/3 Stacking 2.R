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

nmod <- 200

for (pct_train_a in c(0.25, 0.50, 0.75)) {
  # Create the Splits.  Set A is for base model training; B is for oos predictions and training the meta model
  set.seed(1)
  nrow_a <- round(nrow(train) * pct_train_a)
  nrow_b <- nrow(train) - nrow_a
  train_a_i <- sample(c(rep(T, times=nrow_a), rep(F, times=nrow_b)))
  train_b_i <- !train_a_i
  train_a <- train[train_a_i, ]
  train_b <- train[train_b_i, ]
  
  # Create the Folds
  set.seed(1)
  folds_a <- vfold_cv(train_a[, cols])
  
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
    size = nmod
  )
  
  set.seed(1)
  xg_res <- xg_wf %>%
    tune_grid(resamples = folds_a,
              grid = xg_grid,
              control = control_grid(save_pred = T))
  
  xg_best <- select_best(xg_res, 'accuracy')
  xg_final_wf <- xg_wf %>% 
    finalize_workflow(xg_best)
  
  
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
                         size = nmod)
  
  set.seed(1)
  rf_res <- rf_wf %>%
    tune_grid(resamples = folds_a,
              grid = rf_grid,
              control = control_grid(save_pred = T))
  
  rf_best <- select_best(rf_res, 'accuracy')
  rf_final_wf <- rf_wf %>% 
    finalize_workflow(rf_best)
  
  
  ## Lasso
  ls_spec <- logistic_reg(
    mode = "classification",
    penalty = tune(),
    mixture = 1) %>%
    set_engine ( "glmnet" )
  
  ls_rec <- rec 
  
  ls_wf <- workflow() %>% add_model(ls_spec) %>% add_recipe(ls_rec)
  
  ls_grid <- tibble(penalty=10^seq(-4, -0.5, length.out = nmod))
  
  set.seed(1)
  ls_res <- ls_wf %>%
    tune_grid(resamples = folds_a,
              grid = ls_grid,
              control = control_grid(save_pred = T))
  
  ls_best <- select_best(ls_res, 'accuracy')
  ls_final_wf <- ls_wf %>% 
    finalize_workflow(ls_best)
  
  # Meta
  set.seed(1)
  folds_b <- vfold_cv(train_b[, cols])
  
  ## XGBoost
  xg_pred <- (xg_final_wf %>% fit(train) %>% predict(test, type='prob'))[[1]]
  xg_fit <- xg_final_wf %>%
    fit_resamples(folds_b, control=control_resamples(save_pred=T))
  xg_oos_pred <- do.call(rbind, lapply(xg_fit$.predictions, function(x){x[, c('.row', '.pred_TRUE')]}))
  xg_oos_pred <- xg_oos_pred[order(xg_oos_pred$.row), ]
  savePredictions(train_b$PassengerId, xg_oos_pred$.pred_TRUE, paste0('xg_oos_b_', pct_train_a))
  
  ## Random Forest
  rf_pred <- (rf_final_wf %>% fit(train) %>% predict(test, type='prob'))[[1]]
  rf_fit <- rf_final_wf %>%
    fit_resamples(folds_b, control=control_resamples(save_pred=T))
  rf_oos_pred <- do.call(rbind, lapply(rf_fit$.predictions, function(x){x[, c('.row', '.pred_TRUE')]}))
  rf_oos_pred <- rf_oos_pred[order(rf_oos_pred$.row), ]
  savePredictions(train_b$PassengerId, rf_oos_pred$.pred_TRUE, paste0('rf_oos_b_', pct_train_a))
    
  ## LASSO
  ls_pred <- (ls_final_wf %>% fit(train) %>% predict(test,type='prob'))[[1]]
  ls_fit <- ls_final_wf %>%
    fit_resamples(folds_b, control=control_resamples(save_pred=T))
  ls_oos_pred <- do.call(rbind, lapply(ls_fit$.predictions, function(x){x[, c('.row', '.pred_TRUE')]}))
  ls_oos_pred <- ls_oos_pred[order(ls_oos_pred$.row), ]
  savePredictions(train_b$PassengerId, ls_oos_pred$.pred_TRUE, paste0('ls_oos_b_', pct_train_a))
  
  xg_oos_pred <- read.csv(paste0('predictions/', 'xg_oos_b_', pct_train_a, '.csv'))
  rf_oos_pred <- read.csv(paste0('predictions/', 'rf_oos_b_', pct_train_a, '.csv'))
  ls_oos_pred <- read.csv(paste0('predictions/', 'ls_oos_b_', pct_train_a, '.csv'))
  oos_pred <- data.frame(PredXG = xg_oos_pred$Transported, PredRF = rf_oos_pred$Transported, PredLS = ls_oos_pred$Transported)
  test_pred <- data.frame(PredXG = xg_pred, PredRF = rf_pred, PredLS = ls_pred)
  meta_train <- cbind(train_b, oos_pred)
  meta_test <- cbind(test, test_pred)
  
  meta_spec <- logistic_reg()
  meta_rec <- recipe(Transported ~ PredXG + PredRF + PredLS, data = meta_train)
  meta_wf <- workflow() %>% add_model(meta_spec) %>% add_recipe(meta_rec)
  meta_final_fit <- meta_wf %>% fit(meta_train)
  
  meta_pred <- (meta_final_fit %>% predict(meta_test, type = 'prob'))$.pred_TRUE > 0.5
  
  savePredictions(ship_imp_nores[ship_imp_nores$Train == 'FALSE', 'PassengerId'], meta_pred, paste0('meta_b_', pct_train_a))
  
  # logical predictors
  oos_pred <- data.frame(PredXG = as.factor(xg_oos_pred$Transported>.5), PredRF = as.factor(rf_oos_pred$Transported>.5), PredLS = as.factor(ls_oos_pred$Transported>.5))
  test_pred <- data.frame(PredXG = as.factor(xg_pred>.5), PredRF = as.factor(rf_pred>.5), PredLS = as.factor(ls_pred>.5))
  meta_train <- cbind(train_b, oos_pred)
  meta_test <- cbind(test, test_pred)
  
  meta_spec <- logistic_reg()
  meta_rec <- recipe(Transported ~ PredXG + PredRF + PredLS, data = meta_train)
  meta_wf <- workflow() %>% add_model(meta_spec) %>% add_recipe(meta_rec)
  meta_final_fit <- meta_wf %>% fit(meta_train)
  
  meta_pred <- (meta_final_fit %>% predict(meta_test, type = 'prob'))$.pred_TRUE > 0.5
  
  savePredictions(ship_imp_nores[ship_imp_nores$Train == 'FALSE', 'PassengerId'], meta_pred, paste0('meta_b_logical_', pct_train_a))
}