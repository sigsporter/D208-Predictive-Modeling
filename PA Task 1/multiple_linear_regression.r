## Author: Stephen E. Porter
## Title: Multiple Linear Regresesion
## Course: WGU D208: Predictive Modeling
## Instructor: Dr. William Sewell
options(warn=-1)

# Libraries
library(tidyverse)
library(broom)
library(ggplot2)
library(fastDummies)
library(caret)
library(car)
library(corrplot)
library(Hmisc)
library(Metrics)
library(cowplot)

# Import CSV as data frame
df <- read.csv(file = 'C:/WGU/D208 Predictive Modeling/PA Task 1/churn_clean.csv')

# Checking for nulls
sapply(df, function(x) sum(is.na(x)))
dim(df)


str(df)

# Renaming unclear columns named Item1 through Item8 for improved readability &
# confirming they have been renamed correctly

df <- df %>%
  rename(
    Response = Item1,
    Fix = Item2,
    Replacement = Item3,
    Reliability = Item4,
    Options = Item5,
    Respectful = Item6,
    Courteous = Item7,
    Listening = Item8
  )

colnames(df)


# Several columns will not be useful in analysis and therefore will be dropped.
to_drop <- c('CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
             'County', 'Zip', 'Lat', 'Lng', 'TimeZone', 'Job')

dfDropped = df[,!(names(df) %in% to_drop)]
str(dfDropped)


# Creating dummy variables for categorical columns
dfReg <- dummy_cols(dfDropped, remove_selected_columns = TRUE)
names(dfReg) <- gsub(" ", "_", names(dfReg))
names(dfReg) <- gsub("-", "_", names(dfReg))
names(dfReg) <- gsub("[()]", "_", names(dfReg))
str(dfReg)
dim(dfReg)

# Split dfReg into training and testing subsets
set.seed(22)
trainId = createDataPartition(dfReg$Tenure, times = 1, p = 0.7, list = FALSE)

dfTrain = dfReg[trainId,]
dfTest = dfReg[-trainId,]


# Normalize training set
dfTrainNorm <- dfTrain

normalize = function(x) {
  result = (x - min(x)) / (max(x) - min(x))
  return(result)
}

for (i in 1:19) {
  dfTrainNorm[i] <- normalize(dfTrainNorm[i])
}

# Normalize testing set
dfTestNorm <- dfTest

for (i in 1:19) {
  dfTestNorm[i] <- normalize(dfTestNorm[i])
}

# Summary Statistics
summary(dfTrainNorm)

# Data Visualizations
plot_grid(
  # plot 1: children
  ggplot(dfTrainNorm, aes(x=Children)) +
    geom_histogram(),
  
  ggplot(dfTrainNorm, aes(x=Income)) +
     geom_histogram(),
  
  ggplot(dfTrainNorm, aes(x=Tenure)) +
    geom_histogram(),
  
  ggplot(dfTrainNorm, aes(x=Children, y=Tenure)) +
    geom_point() +
    geom_smooth(),
  
  ggplot(dfTrainNorm, aes(x=Income, y=Tenure)) +
    geom_point() +
    geom_smooth(),
  
  ggplot(dfTrainNorm, aes(x=Bandwidth_GB_Year, y=Tenure)) +
    geom_point() +
    geom_smooth(),

  nrow = 2, ncol = 3
)

# Export prepared data sets
write.csv(dfTrainNorm, "C:\\WGU\\D208 Predictive Modeling\\PA Task 1\\D208_dfTrainNorm.csv", row.names = FALSE)
write.csv(dfTestNorm, "C:\\WGU\\D208 Predictive Modeling\\PA Task 1\\D208_dfTestNorm.csv", row.names = FALSE)

# Initial Model
trainColsDf = dfTrainNorm[, names(dfTrainNorm) != "Tenure"]
trainColNames <- colnames(trainColsDf)

predictorVars <- paste(trainColNames, collapse = " + ")

linRegForm <- paste("Tenure ~ ", predictorVars, sep = "")

initialModel <- lm(formula = linRegForm, data = dfTrainNorm)
initialModel
summary(initialModel)
par(mfrow = c(2,2))
plot(initialModel)

# vif(initialModel) # This gives an error that stops the code. Research revealed
# that too much multicollinearity is the cause.
# Solution: remove redundant columns & re-run the model.

initialModelDf <- tidy(initialModel)

# Several columns are labeled NA because of high correlation to another. 
# They will be removed.
initialModelNACols <- initialModelDf$term[is.na(initialModelDf$p.value)]

# Recreate training dataframe & linear model without these column(s)
dfTrainNorm2 <- dfTrainNorm[, !(names(dfTrainNorm)) %in% initialModelNACols]

colNames2 <- colnames(dfTrainNorm2[, names(dfTrainNorm2) != "Tenure"])

predictorVars2 <- paste(colNames2, collapse = " + ")

linRegForm2 <- paste("Tenure ~ ", predictorVars2, sep = "")

initialModel2 <- lm(formula = linRegForm2, data = dfTrainNorm2)

vif(initialModel2) # This will run

# Still too much multicollinearity. Review Correlation Matrix to determine which
# columns to remove

# Custom function found to improve readability of large correlation matrix
# source: http://www.sthda.com/english/wiki/correlation-matrix-a-quick-start-guide-to-analyze-format-and-visualize-a-correlation-matrix-using-r-software

# ++++++++++++++++++++++++++++
# flattenCorrMatrix
# ++++++++++++++++++++++++++++
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

dfTrainNorm2RCor <- rcorr(as.matrix(dfTrainNorm2))
flatCor <- flattenCorrMatrix(dfTrainNorm2RCor$r, dfTrainNorm2RCor$P)

# Create a subset of flatCor that have significant correlation values
highCorrDf <- flatCor[flatCor$cor > 0.75 | flatCor$cor < -0.75 , ]
highCorrDf


# Remove Gender_Male; I am choosing to keep Bandwidth_GB_Year because it is
# highly correlated to Tenure only and Tenure is the dependent variable for
# this model.

# Recreate traning dataframe and linear model without high correlation column(s)
dfTrainNorm3 <- dfTrainNorm2[, names(dfTrainNorm2) != "Gender_Male"]

colNames3 <- colnames(dfTrainNorm3[, names(dfTrainNorm3) != "Tenure"])

predictorVars3 <- paste(colNames3, collapse = " + ")

linRegForm3 <- paste("Tenure ~ ", predictorVars3, sep = "")

initialModel3 <- lm(formula = linRegForm3, data = dfTrainNorm3)

vifScores3 <- vif(initialModel3)


# Now that correlation has been reduced, I will analyze VIF scores. Those with a
# value above 5 will be removed.

# Convert to vifScores to a dataframe for easier manipulation
vifDf3 <- as.data.frame(vifScores3)
vifDf3 <- cbind(dv = rownames(vifDf3), vifDf3)
rownames(vifDf3) <- 1:nrow(vifDf3)
vifDf3

highVif3 <- vifDf3[vifDf3$vifScores3 > 5, ]
highVif3

highVifColNames = highVif3$dv

# Remove columns with high VIF values to improve the model
dfTrainNorm4 <- dfTrainNorm3[, !(names(dfTrainNorm3)) %in% highVifColNames]

colNames4 <- colnames(dfTrainNorm4[, names(dfTrainNorm4) != "Tenure"])

predictorVars4 <- paste(colNames4, collapse = " + ")

linRegForm4 <- paste("Tenure ~ ", predictorVars4, sep = "")

initialModel4 <- lm(formula = linRegForm4, data = dfTrainNorm4)

vifScores4 <- vif(initialModel4)

vifDf4 <- as.data.frame(vifScores4)
vifDf4 <- cbind(dv = rownames(vifDf4), vifDf4)
rownames(vifDf4) <- 1:nrow(vifDf4)
vifDf4

highVif4 <- vifDf4[vifDf4$vifScores4 > 5, ]
highVif4


# There are no VIF values above 5 remaining. This means there is confidence in
# the p-values from initialModel4

summary(initialModel4)
par(mfrow = c(2,2))
plot(initialModel4)

# Those that are noteworthy have p-values <0.05
initModel4Df <- tidy(initialModel4)
options(scipen = 999)

# Selecting useful predictor variables
reducedModelCols <- initModel4Df$term[initModel4Df$p.value < 0.05 ]

# Removing "Intercept"
reducedModelCols <- reducedModelCols[reducedModelCols != "(Intercept)"]

# Creating a reduced model
reducedPredictorVars <- paste(reducedModelCols, collapse = " + ")

reducedLinRegForm <- paste("Tenure ~ ", reducedPredictorVars, sep = "")

reducedModel <- lm(formula = reducedLinRegForm, data = dfTrainNorm)

# Check for VIF
vif(reducedModel)

# View standard plots
par(mfrow = c(2,2))
plot(reducedModel)

# Check for correlation
reducedCorMatrix <- cor(dfTrainNorm[, names(dfTrainNorm) %in% reducedModelCols])
par(mfrow = c(1,1))
corrplot(reducedCorMatrix, 
          method = 'square', 
          type = 'upper', 
          tl.cex = .7, 
          diag = FALSE, 
          order = 'original')


# View residuals by individual predictor variables
plot_grid(
  # plot 1: children
  ggplot(dfTrainNorm, aes(x=Children, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # plot 2: Age
  ggplot(dfTrainNorm, aes(x=Age, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # plot 3: Bandwidth
  ggplot(dfTrainNorm, aes(x=Bandwidth_GB_Year, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 4: Gender
  ggplot(dfTrainNorm, aes(x=Gender_Female, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 5: Churn
  ggplot(dfTrainNorm, aes(x=Churn_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 6: Techie
  ggplot(dfTrainNorm, aes(x=Techie_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 7: Contract
  ggplot(dfTrainNorm, aes(x=Contract_Month_to_month, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 8: Tablet
  ggplot(dfTrainNorm, aes(x=Tablet_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 9: Internet Service
  ggplot(dfTrainNorm, aes(x=InternetService_DSL, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 10: Multiple
  ggplot(dfTrainNorm, aes(x=Multiple_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 11: Online Security
  ggplot(dfTrainNorm, aes(x=OnlineSecurity_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 12: Online Backup
  ggplot(dfTrainNorm, aes(x=OnlineBackup_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 13: Device Protection
  ggplot(dfTrainNorm, aes(x=DeviceProtection_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 14: Paperless Billing
  ggplot(dfTrainNorm, aes(x=PaperlessBilling_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  
  ncol = 7, nrow = 2
)

# Checking Prediction Data accuracy
summary(reducedModel) # All predictor variables are significant; R2 = 0.9961

predictionData <- predict(reducedModel, newdata = dfTestNorm, type = "response")
rmse = rmse(dfTestNorm$Tenure, predictionData) # RMSE = 0.02409544
