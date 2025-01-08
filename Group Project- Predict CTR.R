# Load required libraries
library(tidyverse)
library(caret)
library(dummies)

# Load the dataset
data <- read.csv("ProjectTrainingData.csv")

# Data Transformation
# Extract 'day_of_week' and 'hour_of_day' from 'hour' column
data$day_of_week <- as.factor(weekdays(as.Date(data$hour, format = "%y%m%d%H")))
data$hour_of_day <- as.numeric(substr(data$hour, 7, 8))
data <- data %>% select(-hour)  # Drop the original 'hour' column

# Encoding Categorical Variables
# One-hot encode low-cardinality variables
low_cardinality_vars <- c("C1", "C15", "banner_pos", "site_category", "app_category", "device_type", "device_conn_type")
data <- dummy.data.frame(data, names = low_cardinality_vars, sep = "_")

# Frequency encoding for high-cardinality variables
high_cardinality_vars <- c("C14", "C16", "C17", "C18", "C19", "C20", "C21", "site_domain", "app_domain", "device_model")
for (var in high_cardinality_vars) {
  freq_table <- table(data[[var]])
  data[[var]] <- as.numeric(freq_table[data[[var]]])
}

# Remove ID-related columns
data <- data %>% select(-id)

# Feature Selection using Backward Elimination
set.seed(123)
full_model <- glm(click ~ ., data = data, family = binomial)
stepwise_model <- step(full_model, direction = "backward")

# Split Data into Training and Validation Sets
set.seed(123)
trainIndex <- createDataPartition(data$click, p = 0.9, list = FALSE)
train_data <- data[trainIndex, ]
validation_data <- data[-trainIndex, ]

# Logistic Regression Model
final_model <- glm(click ~ ., data = train_data, family = binomial)

# Predict on Validation Set
pred_probs <- predict(final_model, newdata = validation_data, type = "response")
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

# Evaluate Model
log_loss <- -mean(validation_data$click * log(pred_probs) + (1 - validation_data$click) * log(1 - pred_probs))
conf_matrix <- confusionMatrix(as.factor(pred_labels), as.factor(validation_data$click))
print(conf_matrix)
print(paste("Log Loss:", log_loss))

# Save the model
saveRDS(final_model, "logistic_regression_model.rds")
