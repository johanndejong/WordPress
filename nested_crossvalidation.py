# Do some imports
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

np.random.seed(123)

# Number of folds for the cross-validation
n_fold = 5
# Values of C to try out with the SVM
CC = 2**np.linspace(np.log2(1e-4), np.log2(10), 21)
# Number of repeats to do in permutation analysis
n_perm = 25

# Generate some classification data
X, y = make_multilabel_classification(n_samples=200, n_features=50, n_classes=2, n_labels=1)
y = y[:,0]

def train_and_validate(X, y, ii_train, ii_test, C=1.0):
    X_train = X[ii_train,:]
    y_train = y[ii_train]
    X_test = X[ii_test,:]
    y_test = y[ii_test]
    # Scale the data
    scaler = StandardScaler()
    # Avoid data leakage: fit scaling parameters using only X_train
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Train an SVM, excluding the fold
    svc = SVC(C=C, kernel='linear', probability=True, class_weight='balanced')
    svc = svc.fit(X_train, y_train)
    # Predict the fold
    yh = svc.predict_proba(X_test)[:,1]
    # Return the AUC under the ROC
    return roc_auc_score(y_test, yh)

def cv(X, y, k, C):
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    aucs = np.array([train_and_validate(X, y, ii_train, ii_test, C)
                     for i, (ii_train, ii_test) in enumerate(kf.split(X, y))])
    return aucs

# Test the performance of a linear SVM with C = 1, in 5-fold cross-validation
aucs = cv(X, y, k=n_fold, C=1.0)

plt.clf()
plt.scatter(np.repeat(1, len(aucs)), aucs)
plt.hlines(np.mean(aucs), xmin=0.75, xmax=1.25, color='gray', linestyle='-', label='Average (C=1)')
plt.ylabel('AUC')
plt.xlim((0, 2))
plt.tick_params(axis='x', bottom=False, labelbottom=False)
plt.savefig('cv_for_generalization_estimation.png')

print("Average AUC for C=1: ", np.mean(aucs))

###########################################################################

# Do the cross-validation for each C in CC
def gridsearch_cv(X, y, k, CC):
    aucs = np.array([cv(X, y, k=k, C=C) for C in CC])
    return aucs

aucs = gridsearch_cv(X, y, k=n_fold, CC=CC)

plt.clf()
plt.scatter(np.repeat(CC, aucs.shape[1]).reshape(aucs.shape), aucs)
plt.plot(CC, np.mean(aucs, axis=1), color='red')
ax = plt.gca()
ax.set_xscale('log')
plt.ylabel('AUC')
plt.xlabel('C')
plt.savefig('cv_for_model_selection.png')

print("Optimistically biased AUC directly from gridsearch CV: ", np.max(np.mean(aucs, axis=1)))

###########################################################################

def permute_and_cv(X, y, k, CC):
    # Randomize the class labels. This should result in
    # random performance, i.e. AUC ~= 0.5
    yp = np.random.permutation(y)

    # Cross-validate for each C in CC, and take the
    # average of the folds as the performance estimate
    aucs = np.mean(gridsearch_cv(X, yp, k=k, CC=CC), axis=1)
    # Return the max AUC across the different Cs
    return np.max(aucs)

aucs = [permute_and_cv(X, y, k=n_fold, CC=CC) for i in np.arange(n_perm)]

plt.clf()
plt.hist(aucs, 6)
plt.axvline(x=0.5, color='gray', linestyle='-', label='Expected AUC for random classifier')
plt.axvline(x=np.mean(aucs), color='gray', linestyle='--', label='Observed average AUC')
plt.xlabel('AUC')
plt.ylabel('Count')
plt.legend()
plt.savefig('cv_randomized.png')

###########################################################################

def gridsearch_ncv(X, y, k, CC):
    def inner(X, y, ii_train, ii_test, k, CC):
        # Do a cross-validation for each C, using only the training data
        aucs = gridsearch_cv(X[ii_train,:], y[ii_train], k, CC)
        # Select the C with the highest AUC
        C = CC[np.argmax(np.mean(aucs, axis=1))]
        # Test this C on the test data
        auc = train_and_validate(X, y, ii_train, ii_test, C)
        # Return the auc
        return auc

    kf = StratifiedKFold(n_splits=k, shuffle=True)
    # For each fold, calculate the AUC on the test data
    aucs = np.array([inner(X, y, ii_train, ii_test, k, CC)
                     for i, (ii_train, ii_test) in enumerate(kf.split(X, y))])
    return aucs

#################################################################

def permute_and_ncv(X, y, k, CC):
    # Randomize the class labels. This should result in
    # random performance, i.e. AUC ~= 0.5
    yp = np.random.permutation(y)
    # Cross-validate for each C in CC, and take the
    # average of the folds as the performance estimate
    aucs = gridsearch_ncv(X, yp, k=k, CC=CC)
    # Return the max AUC across the different Cs
    return np.mean(aucs)

aucs = [permute_and_ncv(X, y, k=n_fold, CC=CC) for i in np.arange(n_perm)]

plt.clf()
plt.hist(aucs, 6)
plt.axvline(x=0.5, color='gray', linestyle='-', label='Expected AUC for random classifier')
plt.axvline(x=np.mean(aucs), color='gray', linestyle='--', label='Observed average AUC')
plt.xlabel('AUC')
plt.ylabel('Count')
plt.legend()
plt.savefig('ncv_randomized.png')

##############################################

aucs = gridsearch_ncv(X, y, k=n_fold, CC=CC)

plt.clf()
plt.scatter(np.repeat(1, len(aucs)), aucs)
plt.hlines(np.mean(aucs), xmin=0.75, xmax=1.25, color='gray', linestyle='-', label='Average (C=1)')
plt.ylabel('AUC')
plt.xlim((0, 2))
plt.tick_params(axis='x', bottom=False, labelbottom=False)
plt.savefig('ncv_for_generalization_estimation.png')

print("AUC from nested cross-validation. AUC = ", np.mean(aucs))
