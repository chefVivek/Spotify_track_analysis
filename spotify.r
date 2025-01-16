# 1. Install and Load Required Packages
install.packages(c("tidyverse", "caret", "randomForest", "corrplot", 
                   "gridExtra", "ggplot2", "reshape2", "DataExplorer"))
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(gridExtra)
library(ggplot2)
library(reshape2)
library(DataExplorer)

# Create a directory to save visualizations
dir.create("spotify_visualizations", showWarnings = FALSE)

# 2. Load the Dataset
spotify_data <-  read.csv("C:/Users/chefz/Downloads/dataset.csv", stringsAsFactors = FALSE)

# 3. Initial Data Exploration
# Basic summary of the dataset
summary_output <- capture.output(summary(spotify_data))
writeLines(summary_output, "spotify_visualizations/data_summary.txt")

# Check for missing values
missing_values <- colSums(is.na(spotify_data))
write.csv(missing_values, "spotify_visualizations/missing_values.csv")

# 4. Data Preprocessing
spotify_data$explicit <- as.factor(spotify_data$explicit)

# 5. Exploratory Data Visualization
numerical_features <- c("popularity", "danceability", "energy", 
                        "loudness", "speechiness", "acousticness", 
                        "instrumentalness", "liveness", "valence", "tempo")

# 5.1 Histograms
png("spotify_visualizations/histograms.png", width = 1200, height = 800)
histogram_plots <- lapply(numerical_features, function(feature) {
  ggplot(spotify_data, aes_string(x = feature)) +
    geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
    labs(title = paste("Histogram of", feature)) +
    theme_minimal()
})
grid.arrange(grobs = histogram_plots, ncol = 3)
dev.off()

# 5.2 KDE Plots
png("spotify_visualizations/kde_plots.png", width = 1200, height = 800)
kde_plots <- lapply(numerical_features, function(feature) {
  ggplot(spotify_data, aes_string(x = feature, fill = "explicit")) +
    geom_density(alpha = 0.5) +
    labs(title = paste("KDE of", feature, "by Explicitness")) +
    theme_minimal()
})
grid.arrange(grobs = kde_plots, ncol = 3)
dev.off()

# 5.3 Correlation Heatmap
png("spotify_visualizations/correlation_heatmap.png", width = 800, height = 800)
correlation_matrix <- cor(spotify_data[numerical_features])
corrplot(correlation_matrix, method = "color", type = "upper")
dev.off()

# 5.4 Box Plots
png("spotify_visualizations/boxplots.png", width = 1200, height = 800)
boxplot_features <- lapply(numerical_features, function(feature) {
  ggplot(spotify_data, aes_string(y = feature)) +
    geom_boxplot(fill = "skyblue") +
    labs(title = paste("Boxplot of", feature)) +
    theme_minimal()
})
grid.arrange(grobs = boxplot_features, ncol = 3)
dev.off()

# 6. Prepare Data for Modeling
features <- c("popularity", "danceability", "energy", 
              "loudness", "speechiness", "acousticness", 
              "instrumentalness", "liveness", "valence", "tempo")

set.seed(123)
train_index <- createDataPartition(spotify_data$explicit, p = 0.7, list = FALSE)
train_data <- spotify_data[train_index, ]
test_data <- spotify_data[-train_index, ]

# 7. Logistic Regression
X_train <- train_data[, features]
y_train <- train_data$explicit
X_test <- test_data[, features]
y_test <- test_data$explicit

class_weights <- table(y_train)
class_weights <- class_weights[2] / class_weights[1]

logistic_model <- glm(
  y_train ~ ., 
  data = data.frame(X_train, y_train), 
  family = binomial(link = "logit"),
  weights = ifelse(y_train == 1, class_weights, 1)
)

logistic_prob <- predict(logistic_model, newdata = X_test, type = "response")
logistic_pred <- ifelse(logistic_prob > 0.5, 1, 0)

logistic_pred <- factor(logistic_pred, levels = c(0, 1))
y_test <- factor(y_test, levels = c(0, 1))

logistic_cm <- confusionMatrix(logistic_pred, y_test)

# Save Logistic Regression results
sink("spotify_visualizations/logistic_regression_summary.txt")
print("Logistic Regression Confusion Matrix:")
print(logistic_cm)
summary(logistic_model)
sink()

# 8. Random Forest
rf_model <- randomForest(x = train_data[, features], 
                         y = train_data$explicit, 
                         ntree = 500, 
                         importance = TRUE)

rf_predictions <- predict(rf_model, newdata = test_data[, features])
rf_cm <- confusionMatrix(rf_predictions, test_data$explicit)

# Save Random Forest results
sink("spotify_visualizations/random_forest_summary.txt")
print("Random Forest Performance:")
print(rf_cm)
sink()

# Feature Importance Plot
png("spotify_visualizations/feature_importance.png", width = 800, height = 600)
importance_df <- data.frame(
  Feature = rownames(rf_model$importance),
  Importance = rf_model$importance[,1]
)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance", 
       x = "Features", 
       y = "Importance") +
  theme_minimal()
dev.off()

# 9. Comparative Performance Visualization
performance_data <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(logistic_cm$overall['Accuracy'], rf_cm$overall['Accuracy']),
  Precision = c(logistic_cm$byClass['Precision'], rf_cm$byClass['Precision']),
  Recall = c(logistic_cm$byClass['Recall'], rf_cm$byClass['Recall'])
)

performance_long <- performance_data %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall), 
               names_to = "Metric", 
               values_to = "Value")

png("spotify_visualizations/model_performance_comparison.png", width = 800, height = 600)
ggplot(performance_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", 
       y = "Score") +
  theme_minimal()
dev.off()
