# do some imports for later
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from sklearn.model_selection import KFold
from sklearn.svm import SVC

# fast row or column t-test for equal sample sizes and equal variance
# the hypothesis test is two-sided
def fast_ttest(X, y, axis=0):
    y_un = np.unique(y)
    X1 = X[y==y_un[0],:]
    X2 = X[y==y_un[1],:]
    spool = np.sqrt( (np.var(X1, axis=axis) + np.var(X2, axis=axis)) / 2 )
    tstat = (np.mean(X1, axis=axis) - np.mean(X2, axis=axis)) / spool / np.sqrt(4 / X.shape[axis])
    df = X.shape[axis] - 2
    p = t.cdf(tstat, df)
    return np.minimum(p, 1 - p) * 2

########################################

def generate_random_data(n_sample=int(1e2), n_feature = int(1e5)):
  # the features
  X = np.random.normal(size=(n_sample, n_feature))
  # the class labels
  y = np.repeat([0, 1], n_sample / 2)
  return X, y

########################################

# Function for calculating the misclassification rate for
# one train-test split
def train_and_validate(X, y, ii_train, ii_test, C=1.0):
    X_train = X[ii_train,:]
    y_train = y[ii_train]
    X_test = X[ii_test,:]
    y_test = y[ii_test]
    # Train an SVM, excluding the fold
    svc = SVC(C=C, kernel='linear', probability=True, class_weight='balanced')
    svc = svc.fit(X_train, y_train)
    # Predict the fold
    yh = svc.predict(X_test)
    # Return the AUC under the ROC
    return np.mean(y_test != yh)

# set the seed
np.random.seed(123)

# generate some data
X, y = generate_random_data()

# Train an SVM to predict the class label, and estimate the misclassification rate
# by cross-validation. Use only those samples that passed the global t-test.
selected = fast_ttest(X, y, axis=0) < 0.1
kf = KFold(n_splits=5, shuffle=True)
ers = np.repeat(0.0, 5)
for i, (ii_train, ii_test) in enumerate(kf.split(X[:,selected], y)):
    ers[i] = train_and_validate(X[:,selected], y, ii_train, ii_test)

plt.clf()
plt.scatter(np.repeat(1, len(ers)), ers, color='red')
plt.hlines(np.mean(ers), xmin=0.75, xmax=1.25, color='gray', linestyle='-', label='Average (C=1)')
plt.ylabel('Misclassification rate')
plt.xlim((0, 2))
plt.ylim((-.1, 1))
plt.tick_params(axis='x', bottom=False, labelbottom=False)
plt.savefig('zero_misclassification_rate.png')

########################################

kf = KFold(n_splits=5, shuffle=True)
ers = np.repeat(0.0, 5)
for i, (ii_train, ii_test) in enumerate(kf.split(X, y)):
    selected = fast_ttest(X[ii_train,:], y[ii_train], axis=0) < 0.1
    ers[i] = train_and_validate(X[:,selected], y, ii_train, ii_test)

plt.clf()
plt.scatter(np.repeat(1, len(ers)), ers, color='red')
plt.hlines(np.mean(ers), xmin=0.75, xmax=1.25, color='gray', linestyle='-', label='Average (C=1)')
plt.ylabel('Misclassification rate')
plt.xlim((0, 2))
plt.ylim((-.1, 1))
plt.tick_params(axis='x', bottom=False, labelbottom=False)
plt.savefig('true_misclassification_rate.png')
