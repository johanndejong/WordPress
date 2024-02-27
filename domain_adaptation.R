# Function to generate random data.
generate_data <- function(n) {

  range_x1 <- 1
  range_x2 <- 1

  # The features.
  x1 <- runif(n, -range_x1, range_x1)
  x2 <- runif(n, -range_x2, range_x2)

  # Generate class labels.
  y <- (x1 < 0 & x2 > 0) + 1

  # Generate source and target labels.
  prob <- (x1 + range_x1) / range_x1 / 2
  s <- 1:n %in% sample(n, n/2, prob = prob^5) + 1

  data.frame(
    x1 = x1,
    x2 = x2,
    y = factor(c("class1", "class2")[y]),
    s = factor(c("source", "target")[s])
  )
}

# Function to fit a logistic regression classifier,
# possibly weighted.
fitLRG <- function(df, weights = rep(1, nrow(df))) {
  # Compute the class weights.
  tab <- 1 / table(df$y)
  # Multiply by the instance weights
  weights <- as.numeric(weights * tab[match(df$y, names(tab))])
  # Fit a logistic regression model on the
  # source class label.
  fit <- coef(glmnet(
    x = as.matrix(df[, c("x1", "x2")]),
    y = df$y,
    lambda = seq(1, 0, -0.01),
    weights = weights,
    family = "binomial"
  ))
  fit[, ncol(fit)]
}

# Function to compute instance weights
compute_instance_weights <- function(df) {
  # Fit a logistic regression model on the
  # source/target indicator.
  fit <- glmnet(
    x = as.matrix(df[, c("x1", "x2")]),
    y = df$s,
    lambda = seq(1, 0, -0.01),
    family = "binomial"
  )
  # For each instance, compute the probability
  # that it came from the target data
  p <- predict(
    fit,
    newx = as.matrix(df[,c("x1", "x2")]),
    type = "response"
  )
  p <- p[, ncol(p)]
  p / (1 - p)
}

# Load a package for fitting logistic regression models.
library(glmnet)

# Set the seed for reproducibility.
set.seed(1)

# Generate some random data.
df <- generate_data(1e3)

# Train an unweighted classifier.
fit_unweighted <- fitLRG(df[df$s == "source",])

# Train a re-weighted classifier:
# 1. Compute the instance weights
weights <- compute_instance_weights(df)
# 2. Train a weighted classifier
fit_reweighted <- fitLRG(
  df[df$s == "source",],
  weights = weights[df$s == "source"]
)