################################################### ################################################### #######
# Neural Networks in R
#This study is carried out with neural networks in a dataset of information about concrete production. The accuracy expected by the company is greater than 90%
################################################### ################################################### #######




# Defining the working directory
getwd()

# Loading the data
concrete <- read.csv("concrete.csv")
view(concrete)
str(concrete)

# Normalization Function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Applying the normalization function to the entire dataset
concrete_norm <- as.data.frame(lapply(concrete, normalize))

# Confirming that the range is between 0 and 1
summary(concrete_norm$strength)

# Comparing with the original
summary(concrete$strength)

# Creating training and testing dataset
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

# training the model
install.packages("neuralnet")
library(neuralnet)

# Neural Network with only ONE hidden layer of neurons
set.seed(12345) 
?neuralnet
concrete_model <- neuralnet(formula = strength ~ cement + slag +
                            ash + water + superplastic + 
                            coarseagg + fineagg + age,
                            data = concrete_train)

print(concrete_model)

# Viewing the created network
plot(concrete_model)

# Evaluating the performance
model_results <- compute(concrete_model, concrete_test[1:8])

# Get the predicted values
predicted_strength <- model_results$net.result

# Examining the correlation of predicted values
?color
color(predicted_strength, concrete_test$strength)

# Optimizing the model
# Increasing the number of hidden layers (2 layers) with 5 and 4 neurons respectively.
set.seed(12345)
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic +
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden = c(5,4) )

# Plot
plot(concrete_model2)

# Evaluating the result
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
color(predicted_strength2, concrete_test$strength)

model_results2

###########################################################################################################
# END
###########################################################################################################




