# Load some libraries
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Load and prepare the breast cancer data
data = load_breast_cancer()
y = data.target
X = data.data
# This classification problem is quite easy. For illustration purposes,
# take only a few relatively weakly correlating variables
r = np.array([np.corrcoef(X[:,i], y)[0,1] for i in np.arange(X.shape[1])])
jj = np.argsort(np.abs(r))[:10]
X = X[:,jj]
# For simplicity, restrict both classes to 200 samples
ii_neg = np.random.choice(np.where(y==0)[0], 200)
ii_pos = np.random.choice(np.where(y==1)[0], 200)
ii = np.concatenate([ii_neg, ii_pos])
X = X[ii,:]
y = y[ii]

def train_and_validate(X, y, ii_train, ii_test, C=1.0):
    X_train = X[ii_train,:]
    y_train = y[ii_train]
    X_test = X[ii_test,:]
    y_test = y[ii_test]
    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Train an SVM, excluding the fold
    svc = SVC(C=C, kernel='linear', probability=True)
    svc = svc.fit(X_train, y_train)
    # Predict the fold
    yh = svc.predict_proba(X_test)[:,1]
    # Return the AUC under the ROC
    return roc_auc_score(y_test, yh)

def cv(X, y, k, C):
    kf = KFold(n_splits=k, shuffle=True)
    aucs = np.array([train_and_validate(X, y, ii_train, ii_test, C)
                     for i, (ii_train, ii_test) in enumerate(kf.split(X))])
    return aucs

# Test the performance of a linear SVM with C = 1, in 5-fold cross-validation
aucs_cv = cv(X, y, k=5, C=1e-2)

plt.clf()
plt.scatter(np.repeat(1, len(aucs_cv)), aucs_cv)
plt.ylabel('AUC')
plt.xlim((0, 2))
plt.tick_params(axis='x', bottom=False, labelbottom=False)
plt.savefig('cv_for_generalization_estimation.png')

###########################################################################

# Do the cross-validation for each C in CC
def gridsearch_cv(X, y, k, CC):
    aucs = np.array([cv(X, y, k=k, C=C) for C in CC])
    return aucs

CC = 2**np.linspace(np.log2(1e-3), np.log2(10), 21)
aucs = gridsearch_cv(X, y, k=5, CC=CC)

plt.clf()
plt.scatter(np.repeat(CC, aucs.shape[1]).reshape(aucs.shape), aucs)
plt.plot(CC, np.mean(aucs, axis=1), color='red')
ax = plt.gca()
ax.set_xscale('log')
plt.ylabel('AUC')
plt.xlabel('C')
plt.savefig('cv_for_model_selection.png')

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

CC = 2**np.linspace(np.log2(1e-3), np.log2(10), 21)
aucs = [permute_and_cv(X, y, k=5, CC=CC) for i in np.arange(25)]

plt.clf()
plt.hist(aucs, 6)
plt.axvline(x=0.5, color='gray', linestyle='-', label='Expected AUC for random classifier')
plt.axvline(x=np.mean(aucs), color='gray', linestyle='--', label='Observed average AUC')
plt.xlabel('AUC')
plt.ylabel('Count')
plt.legend()
plt.savefig('cv_randomized.png')

###########################################################################

def ncv(X, y, k, CC):
    def inner(X, y, ii_train, ii_test, k, CC):
        # Do a cross-validation for each C, using only the training data
        aucs = gridsearch_cv(X[ii_train,:], y[ii_train], k, CC)
        # Select the C with the highest AUC
        C = CC[np.argmax(np.mean(aucs, axis=1))]
        # Test this C on the test data
        auc = train_and_validate(X, y, ii_train, ii_test, C)
        # Return the auc
        return auc

    kf = KFold(n_splits=k, shuffle=True)
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
    aucs = ncv(X, yp, k=k, CC=CC)
    # Return the max AUC across the different Cs
    return np.mean(aucs)

CC = 2**np.linspace(np.log2(1e-3), np.log2(10), 21)
aucs = [permute_and_ncv(X, y, k=5, CC=CC) for i in np.arange(25)]

plt.clf()
plt.hist(aucs, 6)
plt.axvline(x=0.5, color='gray', linestyle='-', label='Expected AUC for random classifier')
plt.axvline(x=np.mean(aucs), color='gray', linestyle='--', label='Observed average AUC')
plt.xlabel('AUC')
plt.ylabel('Count')
plt.legend()
plt.savefig('ncv_randomized.png')

##############################################

CC = 2**np.linspace(np.log2(1e-3), np.log2(10), 21)
aucs_cv = cv(X, y, k=5, C=1e-2)
aucs_ncv = ncv(X, y, k=5, CC=CC)
aucs = np.concatenate([aucs_cv, aucs_ncv])

plt.clf()
x = np.repeat([1, 2], len(aucs)/2)
plt.scatter(x, aucs)
plt.hlines(np.mean(aucs_cv), xmin=0.75, xmax=1.25, color='gray', linestyle='-', label='Average (C=1)')
plt.hlines(np.mean(aucs_ncv), xmin=1.75, xmax=2.25, color='gray', linestyle='--', label='Average (Optimized by NCV)')
plt.ylabel('AUC')
plt.xlim((0, 3))
plt.xticks([1, 2], ['C=0.01', 'Optimized by NCV'])
plt.legend()
plt.savefig('ncv_for_generalization_estimation.png')
