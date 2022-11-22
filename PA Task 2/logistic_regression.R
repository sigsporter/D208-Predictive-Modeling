## Author: Stephen E. Porter
## Title: Logistic Regression
## Course: WGU D208: Predictive Modeling
## Instructor: Dr. William Sewell
options(warn=-1)

# Libraries
library(tidyverse)
library(broom)
library(caret)
library(ggplot2)
library(fastDummies)
library(car)
library(Hmisc)
library(corrplot)
library(cowplot)

# Import CSV as data frame
df <- read.csv(file = 'C:/WGU/D208 Predictive Modeling/PA Task 2/churn_clean.csv')

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
to_drop <- c('CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City',
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
trainId = createDataPartition(dfReg$Churn_Yes, times = 1, p = 0.7, list = FALSE)

dfTrain = dfReg[trainId,]
dfTest = dfReg[-trainId,]

# Summary Statistics
summary(dfTrain)

# Data Visualizations
plot_grid(
  ggplot(dfTrain, aes(x=Bandwidth_GB_Year)) +
    geom_histogram(),
  
  ggplot(dfTrain, aes(x=Contacts)) +
    geom_histogram(),
  
  ggplot(dfTrain, aes(x=MonthlyCharge, y=Churn_Yes)) +
    geom_point() +
    stat_smooth(method = "glm", se=FALSE, method.args = list(family=binomial)),
  
  ggplot(dfTrain, aes(x=Tenure, y=Churn_Yes)) +
    geom_point() +
    stat_smooth(method = "glm", se=FALSE, method.args = list(family=binomial)),
  
  
  ncol = 2, nrow = 2
)

# Export prepared data sets
write.csv(dfTrain, "C:\\WGU\\D208 Predictive Modeling\\PA Task 2\\D208_dfTrain.csv", row.names = FALSE)
write.csv(dfTest, "C:\\WGU\\D208 Predictive Modeling\\PA Task 2\\D208_dfTest.csv", row.names = FALSE)

# Initial Model [Note the "Yes" and "No" columns are redundant, so only one will
# be used]
trainColsDf = dfTrain[, -which(names(dfTrain) %in% c("Churn_Yes", "Churn_No"))]
trainColNames <- colnames(trainColsDf)

predictorVars <- paste(trainColNames, collapse = " + ")

logRegForm <- paste("Churn_Yes ~ ", predictorVars, sep = "")

initialModel <- glm(formula = logRegForm, family = "binomial", data = dfTrain)
initialModel

summary(initialModel)
par(mfrow = c(2,2))
plot(initialModel)

# Solution: remove redundant columns & re-run the model.

initialModelDf <- tidy(initialModel)

# Several columns are labeled NA because of high correlation to another. 
# They will be removed.
initialModelNACols <- initialModelDf$term[is.na(initialModelDf$p.value)]

# Recreate training dataframe & linear model without these column(s)
dfTrain2 <- dfTrain[, -which(names(dfTrain) %in% initialModelNACols)]

colNames2 <- colnames(dfTrain2[, -which(names(dfTrain2) %in% c("Churn_Yes", "Churn_No"))])

predictorVars2 <- paste(colNames2, collapse = " + ")

logRegForm2 <- paste("Churn_Yes ~ ", predictorVars2, sep = "")

initialModel2 <- glm(formula = logRegForm2, family = "binomial", data = dfTrain2)

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

dfTrain2RCor <- rcorr(as.matrix(dfTrain2))
flatCor <- flattenCorrMatrix(dfTrain2RCor$r, dfTrain2RCor$P)

# Create a subset of flatCor that have significant correlation values
highCorrDf <- flatCor[flatCor$cor > 0.75 | flatCor$cor < -0.75 , ]
highCorrDf


# Remove Gender_Female & Tenure

# Recreate traning dataframe and linear model without high correlation column(s)
dfTrain3 <- dfTrain2[, -which(names(dfTrain2) %in% c("Gender_Female", "Tenure"))]

colNames3 <- colnames(dfTrain3[, -which(names(dfTrain3) %in% c("Churn_Yes", "Churn_No"))])

predictorVars3 <- paste(colNames3, collapse = " + ")

logRegForm3 <- paste("Churn_Yes ~ ", predictorVars3, sep = "")

initialModel3 <- glm(formula = logRegForm3, family = "binomial", data = dfTrain3)

vifScores3 <- vif(initialModel3)
vifScores3

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
dfTrain4 <- dfTrain3[, !(names(dfTrain3)) %in% highVifColNames]

colNames4 <- colnames(dfTrain4[, -which(names(dfTrain4) %in% c("Churn_Yes", "Churn_No"))])

predictorVars4 <- paste(colNames4, collapse = " + ")

logRegForm4 <- paste("Churn_Yes ~ ", predictorVars4, sep = "")

initialModel4 <- glm(formula = logRegForm4, family = "binomial", data = dfTrain4)

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
reducedModelCols

# Creating a reduced model
reducedPredictorVars <- paste(reducedModelCols, collapse = " + ")

reducedLogRegForm <- paste("Churn_Yes ~ ", reducedPredictorVars, sep = "")

reducedModel <- glm(formula = reducedLogRegForm, family = "binomial", data = dfTrain)

# Check for VIF
vif(reducedModel)

# View standard plots
par(mfrow = c(2,2))
plot(reducedModel)

# Check for correlation
reducedCorMatrix <- cor(dfTrain[, names(dfTrain) %in% reducedModelCols])
par(mfrow = c(1,1))
corrplot(reducedCorMatrix, 
         method = 'square', 
         type = 'upper', 
         tl.cex = .7, 
         diag = FALSE, 
         order = 'original')

# View residuals by individual predictor variables
plot_grid(
  # plot 1: Bandwidth
  ggplot(dfTrain, aes(x=Bandwidth_GB_Year, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # plot 2: State
  ggplot(dfTrain, aes(x=State_RI, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # plot 3: Gender
  ggplot(dfTrain, aes(x=Gender_Male, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 4: Techie
  ggplot(dfTrain, aes(x=Techie_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 5: Contract
  ggplot(dfTrain, aes(x=Contract_Month_to_month, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 6: Internet
  ggplot(dfTrain, aes(x=InternetService_DSL, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 7: Online Backup
  ggplot(dfTrain, aes(x=OnlineBackup_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 8: Device Protection
  ggplot(dfTrain, aes(x=DeviceProtection_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 9: Device Protection
  ggplot(dfTrain, aes(x=TechSupport_No, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),
  # Plot 10: Payment Method
  ggplot(dfTrain, aes(x=PaymentMethod_Electronic_Check, y=residuals(reducedModel))) +
    geom_point() +
    geom_smooth(),

  ncol = 5, nrow = 2
)

# Checking Prediction Data accuracy
summary(reducedModel) # All predictor variables are significant

dfTest$predictionData <- predict(reducedModel, newdata = dfTest, type = "response")

actual_response <- dfTest$Churn_Yes
predicted_response <- round(dfTest$predictionData)
outcomes <- table(predicted_response, actual_response)

confusion <- confusionMatrix(outcomes)
confusion
