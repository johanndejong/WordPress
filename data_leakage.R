# fast row t-test for equal sample sizes and equal variance
# the hypothesis test is two-sided
row_ttest <- function(X, y) {
  library(matrixStats)
  X1 <- X[, y == levels(y)[1]]
  X2 <- X[, y == levels(y)[2]]
  spool <- sqrt( (rowVars(X1) + rowVars(X2)) / 2 )
  tstat <- (rowMeans(X1) - rowMeans(X2)) / spool / sqrt(4 / ncol(X))
  df <- ncol(X) - 2
  p <- pt(tstat, df)
  pmin(p, 1 - p) * 2
}

########################################

generate_random_data <- function(
  # number of samples in the random data
  nsamples = 1e2,
  # number of features in the random data
  nfeatures = 1e5
) {
  # the features
  X <- matrix(
    rnorm(nsamples * nfeatures),
    nrow = nfeatures,
    ncol = nsamples
  )
  # the class labels
  y <- gl(2, nsamples / 2)
  list(
    X = X,
    y = y
  )
}

########################################

# load some libraries
library(caret)
library(e1071)

# set the seed
set.seed(123)

# generate some data
Xy <- generate_random_data()
X <- Xy$X
y <- Xy$y

# apply the t-test filter
selected <- row_ttest(X, y) < 0.1

# Train an SVM to predict the class label,
# and estimate the misclassification rate by
# cross-validation.
folds <- createFolds(y, k = 5)
cv <- unlist(lapply(folds, function(fold) {
  # train and test a model
  fit <- svm(x = t(X[selected, -fold]), y = y[-fold])
  pred <- predict(fit, newdata = t(X[selected, fold]))
  ct <- table(pred, y[fold])
  # get the misclassification rate
  1 - sum(diag(ct)) / sum(ct)
}))
barplot(
  cv, ylim = c(0, 1), las = 2,
  ylab = "misclassification rate"
)

########################################

# load some libraries
library(caret)
library(e1071)

# set the seed
set.seed(123)

# generate some data
Xy <- generate_random_data()
X <- Xy$X
y <- Xy$y

# Train an SVM to predict the class label,
# and estimate the misclassification rate by
# cross-validation.
folds <- createFolds(y, k = 5)
cv <- unlist(lapply(folds, function(fold) {
  # apply the t-test filter within the cross-validation loop!
  selected <- row_ttest(X[,-fold], y[-fold]) < 0.1
  # train and test a model
  fit <- svm(x = t(X[selected, -fold]), y = y[-fold])
  pred <- predict(fit, newdata = t(X[selected, fold]))
  ct <- table(pred, y[fold])
  # get the misclassification rate
  mean(pred != y[fold])
}))
barplot(
  cv, ylim = c(0, 1), las = 2,
  ylab = "misclassification rate"
)
