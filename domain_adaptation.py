import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm

# Function to generate random data.
def generate_data(n):

    range_x = [-1, 1]

    # The features.
    X = np.random.uniform(range_x[0], range_x[1], (n, 2))

    # Generate class labels.
    y = ((X[:,0] < 0) & (X[:,1] > 0)).astype('int')

    # Generate source and target labels.
    is_target = np.random.uniform(size=X.shape[0]) < norm.cdf(X[:, 0], loc=0, scale=0.2)
    is_target = is_target.astype('int')

    return X, y, is_target

# Function to fit a logistic regression classifier, possibly instance-weighted.
def fitLRG(X, y, weights=None):
    if weights is None:
        weights = np.ones(len(y))

    # Compute the class weights.
    tab = 1 / np.unique(y, return_counts=True)[1]
    # Multiply by the instance weights
    weights = weights * tab[y]
    # Fit a logistic regression model on the source class label.
    fit = LogisticRegression(penalty=None).fit(X, y, sample_weight=weights)
    c = fit.intercept_[0]
    w1, w2 = fit.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    b = -c / w2
    a = -w1 / w2
    return lambda x: a*x + b


# Function to compute instance weights
def compute_instance_weights(X, is_target):

    # Fit a logistic regression model on the source/target indicator.
    fit = LogisticRegression(class_weight='balanced', penalty=None).fit(X, is_target)

    # For each instance, compute the probability that it came from the target data
    p = fit.predict_proba(X)[:,1]

    # return the ratio
    return p / (1 - p)

# Set the seed for reproducibility.
np.random.seed(111)

# Generate some random data.
X, y, is_target = generate_data(int(1e3))

# Train an unweighted classifier on the source data.
mask = is_target == 0
fit_source_unweighted = fitLRG(X[mask,:], y[mask])

# Train a re-weighted classifier on the source data:
# 1. Compute the instance weights
weights = compute_instance_weights(X, is_target)

# 2. Train a weighted classifier
mask = is_target == 0
fit_source_reweighted = fitLRG(X[mask,:], y[mask], weights[mask])

# For reference, train a classifier on the target data directly
mask = is_target == 1
fit_target = fitLRG(X[mask,:], y[mask])

####################
# Do some plotting #
####################

# Model misspecification
plt.clf()
plt.scatter(X[y==0, 0], X[y==0, 1], marker='o', color='red', label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], marker='o', color='lightblue', label='Class 1')
f = fitLRG(X, y)
x = np.array([-1, 1])
plt.plot(x, f(x), linestyle='--', color='black', linewidth=3.0)
plt.hlines(y=0, xmin=-1, xmax=1, color='lightgray', linestyle='--')
plt.vlines(x=0, ymin=-1, ymax=1, color='lightgray', linestyle='--')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.title('Model misspecification')
plt.savefig('model_misspecification.png')

# Data without target class labels
plt.clf()
plt.scatter(X[(y==0) & (is_target==0), 0], X[(y==0) & (is_target==0), 1], marker='o', color='red', label='Source data: class 0')
plt.scatter(X[(y==1) & (is_target==0), 0], X[(y==1) & (is_target==0), 1], marker='o', color='lightblue', label='Source data: class 1')
plt.scatter(X[is_target==1, 0], X[is_target==1, 1], marker='^', color='lightgray', label='Target data: class unknown')
plt.hlines(y=0, xmin=-1, xmax=1, color='lightgray', linestyle='--')
plt.vlines(x=0, ymin=-1, ymax=1, color='lightgray', linestyle='--')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.title('Data without target class labels')
plt.savefig('data_without_target_class_labels.png')

# Optimal decision boundary for the source data
plt.clf()
plt.scatter(X[(y==0) & (is_target==0), 0], X[(y==0) & (is_target==0), 1], marker='o', color='red', label='Source data: class 0')
plt.scatter(X[(y==1) & (is_target==0), 0], X[(y==1) & (is_target==0), 1], marker='o', color='lightblue', label='Source data: class 1')
x = np.array([-1, 1])
plt.plot(x, fit_source_unweighted(x), linestyle='--', color='black', linewidth=3.0)
plt.hlines(y=0, xmin=-1, xmax=1, color='lightgray', linestyle='--')
plt.vlines(x=0, ymin=-1, ymax=1, color='lightgray', linestyle='--')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.title('Optimal decision boundary\nfor the source data')
plt.savefig('source_data__source_boundary.png')

# Target data, with true labels, and source data decision boundary
plt.clf()
plt.scatter(X[(y==0) & (is_target==1), 0], X[(y==0) & (is_target==1), 1], marker='^', color='red', label='Target data: true class 0')
plt.scatter(X[(y==1) & (is_target==1), 0], X[(y==1) & (is_target==1), 1], marker='^', color='lightblue', label='Target data: true class 1')
x = np.array([-1, 1])
plt.plot(x, fit_source_unweighted(x), linestyle='--', color='black', linewidth=3.0)
plt.hlines(y=0, xmin=-1, xmax=1, color='lightgray', linestyle='--')
plt.vlines(x=0, ymin=-1, ymax=1, color='lightgray', linestyle='--')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.title('Target data, with true labels,\nand source data decision boundary')
plt.savefig('target_data__source_boundary.png')
plt.title('Unweighted training on the source data')
plt.savefig('unweighted_training_on_sourcedata.png')

# Optimal decision boundary for the target data
plt.clf()
plt.scatter(X[(y==0) & (is_target==1), 0], X[(y==0) & (is_target==1), 1], marker='^', color='red', label='Target data: true class 0')
plt.scatter(X[(y==1) & (is_target==1), 0], X[(y==1) & (is_target==1), 1], marker='^', color='lightblue', label='Target data: true class 1')
x = np.array([-1, 1])
plt.plot(x, fit_target(x), linestyle='--', color='black', linewidth=3.0)
plt.hlines(y=0, xmin=-1, xmax=1, color='lightgray', linestyle='--')
plt.vlines(x=0, ymin=-1, ymax=1, color='lightgray', linestyle='--')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.title('Optimal decision boundary\nfor the target data')
plt.savefig('optimal_decision_boundary_target_data.png')

# Re-weighted training on source data
plt.clf()
plt.scatter(X[(y==0) & (is_target==1), 0], X[(y==0) & (is_target==1), 1], marker='^', color='red', label='Target data: true class 0')
plt.scatter(X[(y==1) & (is_target==1), 0], X[(y==1) & (is_target==1), 1], marker='^', color='lightblue', label='Target data: true class 1')
x = np.array([-1, 1])
plt.plot(x, fit_source_reweighted(x), linestyle='--', color='black', linewidth=3.0)
plt.hlines(y=0, xmin=-1, xmax=1, color='lightgray', linestyle='--')
plt.vlines(x=0, ymin=-1, ymax=1, color='lightgray', linestyle='--')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.title('Re-weighted training on source data')
plt.savefig('reweighted_training_on_source_data.png')
