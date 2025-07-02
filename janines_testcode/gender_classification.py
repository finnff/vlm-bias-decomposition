import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


#make classifier
def train_gender_classifier(clip_embeddings, gender_labels):

    if hasattr(clip_embeddings, "numpy"):
        X = clip_embeddings.numpy()
    else:
        X = clip_embeddings

    if hasattr(gender_labels, "numpy"):
        y = gender_labels.numpy()
    else:
        y = gender_labels


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    misclassified = (y_pred != y_test)

    # Get image indices from test set
    misclassified_indices = np.where(misclassified)[0]
    return clf, X_train, X_test, y_train, y_test, y_pred, misclassified_indices


def eval_classifier(y_test, y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test):
    print(f"Gender classifier training accuracy: {clf.score(X_train, y_train):.4f}")
    print(f"Gender classifier test accuracy:     {clf.score(X_test, y_test):.4f}")
    return  clf.score(X_test, y_test)
