import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# -- Pre-split globally once: call this at the top level, once.
def global_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

# -- New fast classifier using SGD
def train_gender_classifier_fast(X_train, y_train, X_test, y_test, 
                                 max_iter=5, tol=1e-3):
    """
    Trains an SGD-based logistic model on the pre-split data.
    X_train, y_train, X_test, y_test come from a single call to global_train_test_split.
    Returns: fitted clf, y_pred on X_test.
    """
    # Use a simple log-loss SGD with partial_fit style internally, n_jobs not supported but it's very fast
    clf = SGDClassifier(
        loss="log_loss",      # logistic regression
        max_iter=max_iter,    # number of epochs
        tol=tol,              # early stopping tolerance
        learning_rate="optimal", 
        random_state=0
    )
    # partial_fit on the full train set for multiple epochs
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

# -- Accuracy wrapper
from sklearn.metrics import accuracy_score
def gender_classifier_accuracy_fast(clf, X_test, y_test):
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Gender classifier test accuracy: {acc:.4f}")
    return acc
