# Load some packages
library(kernlab)
library(caret)
library(mlbench)
library(PRROC)

# Set the random seed for reproducibility
set.seed(111)

# Load and prepare the breast cancer data
data(BreastCancer)
data <- BreastCancer[!is.na(BreastCancer$Bare.nuclei), -1]
# For simplicity, restrict both classes to 200 samples
data <- data[
  c(
    sample(which(data$Class == levels(data$Class)[1]), 200),
    sample(which(data$Class == levels(data$Class)[2]), 200)
  ),
]
y <- data$Class
X <- data
X$Class <- NULL

# Test the performance of a linear SVM with C = 1
folds <- createFolds(data$Class, k = 5)
# For each fold ...
auc <- sapply(folds, function(fold) {
  # Train an SVM, excluding the fold
  fit <- ksvm(
    Class ~ .,
    data = data[-fold,],
    kernel = "vanilladot",
    kpar = list(),
    C = 1,
    prob.model = TRUE,
    Class.weights = 1 / table(data$Class[-fold])
  )
  # Predict the fold
  yh <- predict(fit, newdata = data[fold,], type = "probabilities")
  # Compare the predictions to the labels
  posneg <- split(yh[,1], data$Class[fold])
  # Return the AUC under the ROC
  roc.curve(posneg[[1]], posneg[[2]])$auc
})

#####################################################

# Function for one round of training and validating an SVM
train_and_validate <- function(
  data,
  fold,
  C
) {
  # Train an SVM, excluding the fold
  fit <- ksvm(
    Class ~ .,
    data = data[-fold,],
    kernel = "vanilladot",
    kpar = list(),
    C = C,
    prob.model = TRUE,
    Class.weights = 1 / table(data$Class[-fold])
  )
  # Predict the fold
  yh <- predict(fit, newdata = data[fold,], type = "probabilities")
  # Compare the predictions to the labels
  posneg <- split(yh[,1], data$Class[fold])
  # Return the AUC under the ROC
  roc.curve(posneg[[1]], posneg[[2]])$auc
}

# Function for doing a k-fold cross-validation for each C in CC
cv <- function(
  data,
  k,
  CC,
  seed = NULL
) {
  # Set the seed, if given
  if (!is.null(seed)) {
    set.seed(seed)
  }
  # For each value of the hyperparameter C ...
  auc <- lapply(CC, function(C) {
    folds <- createFolds(data$Class, k = k)
    # For each fold ...
    sapply(folds, function(fold) {
      # Train an SVM, and validate on the fold
      train_and_validate(
        data,
        fold,
        C
      )
    })
  })
  auc
}

# Do the cross-validation for each C in CC
auc <- cv(
  data = data,
  k = 5,
  CC = 2^seq(log2(.01), log2(10), length.out = 21),
  seed = 111
)

#####################################################

set.seed(111)
auc <- replicate(25, {
  # Randomize the class labels. This should result in
  # random performance, i.e. AUC ~= 0.5
  data$Class <- sample(data$Class)

  # Cross-validate for each C in CC, and take the
  # average of the folds as the performance estimate
  auc <- sapply(cv(
    data = data,
    k = 5,
    CC = 2^seq(log2(.01), log2(10), length.out = 11)
  ), mean)
  # Take the max AUC across the different Cs
  max(auc)
})

#####################################################

ncv <- function(
  data,
  k,
  CC,
  seed = NULL
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  folds <- createFolds(data$Class, k = k)
  # For each fold ...
  auc <- sapply(folds, function(fold) {
    # Do a cross-validation for each C
    auc <- cv(
      data[-fold,],
      k,
      CC,
      seed = seed
    )
    # Select the C with the highest AUC
    C <- CC[which.max(sapply(auc, mean))]
    C  1, sample(C, 1), C)
    # Test this C on the test data
    train_and_validate(
      data,
      fold = fold,
      C = C
    )
  })
  auc
}

#####################################################

set.seed(111)
auc <- replicate(25, {
  cat(".")
  # Randomize the class labels. This should result in
  # random performance, i.e. AUC ~= 0.5
  data$Class <- sample(data$Class)

  # This returns k scores
  auc <- ncv(
    data = data,
    k = 5,
    CC = 2^seq(log2(.01), log2(10), length.out = 11)
  )
  # Take the average as the performance estimate
  mean(auc)
})

#####################################################

# test on BreastCancer data
auc <- ncv(
  data = data,
  k = 5,
  CC = 2^seq(log2(.01), log2(10), length.out = 21),
  seed = 111
)

#####################################################
