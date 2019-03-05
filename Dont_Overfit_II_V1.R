library(tidyverse)
library(reshape2)
library(ggcorrplot)
library(recipes)
library(caret)
library(MLmetrics)
library(keras)

set.seed(865)
setwd("C:/Users/Ryan/Desktop/Data Science/Kaggle Competitions/Don't Overfit II")

# -----------------------------------------------------------------------------------
# Load in Data and Inspect
# -----------------------------------------------------------------------------------

# Training Data
train <- read_csv("train.csv", 
                  col_types = cols(id = col_character(), 
                                   target = col_character()))
# Testing Data
test <- read_csv("test.csv", 
                 col_types = cols(id = col_character()))

train$target <- ifelse(train$target == '1.0', '1', '0')

# Look at Data
glimpse(train)
glimpse(test)

# Store Test Set IDs
test_ids <- test$id

# Remove ID Variables
train$id <- NULL
test$id <- NULL

# Combine Training and Testing
train$train_test <- 'train'
test$train_test <- 'test'
dat <- bind_rows(train, test)



# -----------------------------------------------------------------------------------
# Exploratory Data Analysis -- Distributions
# -----------------------------------------------------------------------------------

# Target distribution
prop.table(table(dat$target)) # 1:2 split

# Select only independent variables
X_dat <- dat %>%
  dplyr::select(-target, -train_test)

# Normality Test for independent variable
p_vals <- c()
for (i in 1:ncol(X_dat)) {
  p_vals[i] <- ks.test(as.matrix(X_dat[, i]), rnorm(20000))$p.value
}

# How many variables are not normal? (p-value less than 0.05)
length(which(p_vals < 0.05))/ncol(X_dat) # 4%

# Mean and sd of each independent variable
boxplot(colMeans(X_dat)) # Roughly zero
boxplot(apply(X_dat, 2, sd)) # Roughly one

# Data are probably all generated using Standard Normal Distribution

variable_weights <- train %>% 
  select(-train_test) %>% 
  mutate(target = as.numeric(target)) %>% 
  cor() %>% 
  round(2) %>% 
  melt() %>% 
  filter(Var1 == 'target') %>% 
  filter(Var2 != 'target') %>% 
  arrange(desc(abs(value))) %>%
  head(19) %>% 
  mutate(wt = -1*(value/sum(value))) %>%
  select(wt) %>%
  pull()

preds <- (variable_weights[1]*train$`33`) + (variable_weights[2]*train$`65`) + (variable_weights[3]*train$`217`) +
  (variable_weights[4]*train$`117`) + (variable_weights[5]*train$`91`) + (variable_weights[6]*train$`24`) +
  (variable_weights[7]*train$`73`) + (variable_weights[8]*train$`295`) + (variable_weights[9]*train$`80`) +
  (variable_weights[10]*train$`183`) + (variable_weights[11]*train$`189`) + (variable_weights[12]*train$`199`) + 
  (variable_weights[13]*train$`194`) + (variable_weights[14]*train$`16`) + (variable_weights[15]*train$`39`) +
  (variable_weights[16]*train$`90`) + (variable_weights[17]*train$`129`) + (variable_weights[18]*train$`133`) +
  (variable_weights[19]*train$`165`)

AUC(preds, train$target)

pred_weights <- (variable_weights[1]*test$`33`) + (variable_weights[2]*test$`65`) + (variable_weights[3]*test$`217`) +
  (variable_weights[4]*test$`117`) + (variable_weights[5]*test$`91`) + (variable_weights[6]*test$`24`) +
  (variable_weights[7]*test$`73`) + (variable_weights[8]*test$`295`) + (variable_weights[9]*test$`80`) +
  (variable_weights[10]*test$`183`) + (variable_weights[11]*test$`189`) + (variable_weights[12]*test$`199`) +
  (variable_weights[13]*test$`194`) + (variable_weights[14]*test$`16`) + (variable_weights[15]*test$`39`) +
  (variable_weights[16]*test$`90`) + (variable_weights[17]*test$`129`) + (variable_weights[18]*test$`133`) +
  (variable_weights[19]*test$`165`)

# Submission Table
submission_weights <- data_frame(ID = test_ids, target = pred_weights)
write.csv(submission_weights, 'dont_overfit_weights_submission.csv', row.names = FALSE)