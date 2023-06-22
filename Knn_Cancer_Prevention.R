###########################################################################################################
#                                                                                                         #
#                                                   #
#                                                                                                         #
#                                     Machine Learning - KNN                                       #
#                                                                                                         #
#                              Predicting Breast Cancer Occurrence                                       #
#                                                                                                         #
###########################################################################################################

setwd("C:/Users/e-gustavo.silva/Downloads/Telegram Desktop")
getwd()

# Business Problem Definition: Breast Cancer Occurrence Prediction

################################################### # ################################################## ## ##
## Step 1 - Collecting the Data
################################################### # ################################################## ## ##

# Breast cancer data includes 569 observations of cancer biopsies,
# each with 32 characteristics (variables). One feature is an identification number (ID),
# another is the cancer diagnosis, and the remaining 30 are numeric laboratory measurements.
# The diagnosis is coded as "M" to indicate malignant or "B" to indicate benign.

data <- read.csv("dataset_cancer.csv", stringsAsFactors = FALSE)
str(data)
View(data)

######################################################################################################
## Step 2 - Preprocessing
######################################################################################################

# Removing the ID column
# Regardless of the machine learning method, ID variables should always be excluded.
# Otherwise, this can lead to incorrect results because the ID can be used to uniquely "predict" each example.
# Consequently, a model that includes an identifier may suffer from overfitting and will be difficult to generalize to other data.

data$id <- NULL

# Adjusting the label of the target variable
data$diagnosis <- sapply(data$diagnosis, function(x) { ifelse(x == 'M', 'Malignant', 'Benign') })

# Many classifiers require variables to be of the Factor type
table(data$diagnosis)
data$diagnosis <- factor(data$diagnosis, levels = c("Benign", "Malignant"), labels = c("Benign", "Malignant"))
str(data$diagnosis)

# Checking the proportions
round(prop.table(table(data$diagnosis)) * 100, digits = 1)

# Measures of Central Tendency
# We detected a scaling issue among the data, which needs to be normalized.
# The distance calculation performed by kNN is dependent on the scale measures in the input data.
summary(data[c("radius_mean", "area_mean", "smoothness_mean")])

# Creating a normalization function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Testing the normalization function - the results should be identical
normalize(c(1, 2, 3, 4, 5))
normalize(c(10, 20, 30, 40, 50))

# Normalizing the data
normalized_data <- as.data.frame(lapply(data[2:31], normalize))
View(normalized_data)

######################################################################################################
## Step 3: Training the KNN Model
######################################################################################################

# Loading the "class" library
# install.packages("class")
library(class)
?knn

# Creating training and test data
train_data <- normalized_data[1:469, ]
test_data <- normalized_data[470:569, ]

# Creating labels for training and test data
train_labels <- data[1:469, 1]
test_labels <- data[470:569, 1]
length(train_labels)
length(test_labels)

# Creating the model
knn_model_v1 <- knn(train = train_data, 
                    test = test_data,
                    cl = train_labels, 
                    k = 21)

# The knn() function returns a factor object with predictions for each example in the test dataset
summary(knn_model_v1)

######################################################################################################
## Step 4: Evaluating and Interpreting the Model
######################################################################################################

# Loading the "gmodels" library
install.packages("gmodels")

library(gmodels)

# Creating a cross-tabulation of predicted vs. actual data
# We will use a sample with 100 observations: length(test_labels)
CrossTable(x = test_labels, y = knn_model_v1, prop.chisq = FALSE)

# Interpreting the Results
# The cross-tabulation table shows 4 possible values, representing true/false positive and negative.
# We have two columns listing the original labels in the observed data.
# We have two rows listing the labels of the test data.

# We have:
# Scenario 1: Benign Cell (Observed) x Benign (Predicted) - 61 cases - true positive
# Scenario 2: Malignant Cell (Observed) x Benign (Predicted) - 00 cases - false positive (model error)
# Scenario 3: Benign Cell (Observed) x Malignant (Predicted) - 02 cases - false negative (model error)
# Scenario 4: Malignant Cell (Observed) x Malignant (Predicted) - 37 cases - true negative

# Reading the Confusion Matrix (Perspective of having or not having the disease):

# True Negative  = Our model predicted that the person did NOT have the disease, and the data showed that they really did NOT have the disease.
# False Positive = Our model predicted that the person had the disease, but the data showed that NO, the person did not have the disease.
# False Negative = Our model predicted that the person did NOT have the disease, but the data showed that YES, the person had the disease.
# True Positive = Our model predicted that the person had the disease, and the data showed that YES, the person had the disease.

# False Positive - Type I Error
# False Negative - Type II Error

# Model Accuracy Rate: 98% (correctly predicted 98 out of 100 cases)

######################################################################################################
## Step 5: Optimizing Model Performance
######################################################################################################

# Using the scale() function to standardize the z-score
?scale()
z_data <- as.data.frame(scale(data[-1]))

# Confirming successful transformation
summary(z_data$area_mean)

# Creating new training and test datasets
train_data <- z_data[1:469, ]
test_data <- z_data[470:569, ]

train_labels <- data[1:469, 1] 
test_labels <- data[470:569, 1]

# Reclassifying
knn_model_v2 <- knn(train = train_data, 
                    test = test_data,
                    cl = train_labels, 
                    k = 47)

# Creating a cross-tabulation of predicted vs. actual data
CrossTable(x = test_labels, y = knn_model_v2, prop.chisq = FALSE)

# Try different values for k

######################################################################################################
## Step 6: Building a Model with Support Vector Machine (SVM) Algorithm
######################################################################################################

# Setting the seed for reproducible results
set.seed(40) 

# Prepare the dataset
data <- read.csv("dataset.csv", stringsAsFactors = FALSE)
data$id <- NULL
data[,'index'] <- ifelse(runif(nrow(data)) < 0.8, 1, 0)
View(data)

# Training and test data
trainset <- data[data$index == 1, ]
testset <- data[data$index == 0, ]

# Get the index
trainColNum <- grep('index', names(trainset))

# Remove the index from the datasets
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]

# Get column index of the target variable in the dataset
typeColNum <- grep('diag', names(data))

# Create the model
# We set the kernel to radial since this dataset does not have a linear plane that can be drawn
library(e1071)
?svm
svm_model_v1 <- svm(diagnosis ~ ., 
                    data = trainset, 
                    type = 'C-classification', 
                    kernel = 'radial') 


# Predictions

# Predictions on the training data
pred_train <- predict(svm_model_v1, trainset) 

# Percentage of correct predictions with the training dataset
mean(pred_train == trainset$diagnosis)  


# Predictions on the test data
pred_test <- predict(svm_model_v1, testset) 

# Percentage of correct predictions with the test dataset
mean(pred_test == testset$diagnosis)  

# Confusion Matrix
table(pred_test, testset$diagnosis)

######################################################################################################
## Step 7: Building a Model with Random Forest Algorithm
######################################################################################################

# Creating the model
library(rpart)
rf_model_v1 <- rpart(diagnosis ~ ., data = trainset, control = rpart.control(cp = .0005)) 

# Predictions on the test data
tree_pred <- predict(rf_model_v1, testset, type = 'class')

# Percentage of correct predictions with the test dataset
mean(tree_pred == testset$diagnosis) 

# Confusion Matrix
table(tree_pred, testset$diagnosis)

######################################################################################################
#                                         END
######################################################################################################


