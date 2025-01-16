# Spotify Dataset Analysis

This project analyzes a Spotify dataset using various machine learning techniques and data visualization methods. The goal is to explore the dataset, visualize key insights, and compare the performance of Logistic Regression and Random Forest models in predicting explicit content.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Steps and Code Breakdown](#steps-and-code-breakdown)
- [Results](#results)
- [Visualizations](#visualizations)
- [Usage](#usage)
- [License](#license)

## Installation

### Required R Packages

Make sure to install the following packages before running the code:

```r
install.packages(c("tidyverse", "caret", "randomForest", "corrplot",
                   "gridExtra", "ggplot2", "reshape2", "DataExplorer"))
```

Load the libraries using:

```r
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(gridExtra)
library(ggplot2)
library(reshape2)
library(DataExplorer)
```

## Dataset

Download the dataset and place it in your working directory. Replace the file path in the code with the location of your dataset:

```r
spotify_data <-  read.csv("C:/Users/chefz/Downloads/dataset.csv", stringsAsFactors = FALSE)
```

## Project Overview

1. **Initial Exploration:**
   - Summarize the dataset.
   - Check for missing values.

2. **Data Preprocessing:**
   - Convert the `explicit` column to a factor.

3. **Data Visualization:**
   - Histograms
   - KDE Plots
   - Correlation Heatmap
   - Box Plots

4. **Modeling:**
   - Logistic Regression
   - Random Forest

5. **Model Comparison:**
   - Accuracy, Precision, and Recall comparison between models.

## Steps and Code Breakdown

1. **Install and Load Required Packages:**
   Install necessary libraries for data analysis and visualization.

2. **Load the Dataset:**
   Load the Spotify dataset and explore its structure.

3. **Data Preprocessing:**
   Handle missing values and transform columns as needed.

4. **Exploratory Data Visualization:**
   Generate visualizations to better understand the data distribution and relationships.

5. **Model Building:**
   Train Logistic Regression and Random Forest models on the dataset and evaluate their performance.

6. **Performance Comparison:**
   Compare model accuracy, precision, and recall using bar plots.

## Results

- **Logistic Regression:**
  - Confusion Matrix and summary saved to `logistic_regression_summary.txt`.

- **Random Forest:**
  - Confusion Matrix and feature importance saved to `random_forest_summary.txt` and `feature_importance.png`.

- **Comparative Performance:**
  - Model performance metrics are visualized in `model_performance_comparison.png`.

## Visualizations

All visualizations are saved in the `spotify_visualizations` folder:

1. Histograms (`histograms.png`)
2. KDE Plots (`kde_plots.png`)
3. Correlation Heatmap (`correlation_heatmap.png`)
4. Box Plots (`boxplots.png`)
5. Feature Importance (`feature_importance.png`)
6. Model Performance Comparison (`model_performance_comparison.png`)

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/chefVivek/Spotify_track_analysis.git
   ```

2. Open the R script and replace the dataset path with your file location.

3. Run the script in RStudio to generate visualizations and model results.

4. Check the `spotify_visualizations` folder for outputs.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as long as proper credit is given.

