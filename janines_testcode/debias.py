import numpy as np
import torch
from clip_utils import get_labels, load_celeba_dataset, get_all_embeddings_and_attrs
from gender_classification import train_gender_classifier, gender_classifier_accuracy, eval_classifier
from attribute_bias_analysis import compare_groups_ttest
import io
import sys


bias_attribute_names = ["Attractive"]

def hard_debias(embeddings, bias_basis):
    """
    Remove the enitre component / embeddings

    subtract full projection P=BB^T
    """
    B = bias_basis.T  #D, k
    P = B @ B.T       #D, D projection onto bias space
    if isinstance(embeddings, np.ndarray):
        return embeddings - embeddings @ P
    else:
        return embeddings - embeddings @ P.to(embeddings.device)


def soft_debias(embeddings, bias_basis, lam=1.0):
    """
    making projection smaller of biased vectors.

    math:
    min_{x'} ||x-x'||^2+lam * ||Proj_B(x')||^2

    """
    B = bias_basis.T        #D, k
    P = B @ B.T             #D, D
    alpha = lam / (1.0 + lam)  #how much smaller param
    if isinstance(embeddings, np.ndarray):
        return embeddings - alpha * (embeddings @ P)
    else:
        return embeddings - alpha * (embeddings @ P.to(embeddings.device))

dataset = load_celeba_dataset(root="./data", split="train", download=False)
clip_embeddings, gender_labels, images = get_labels(dataset, batch_size=64, max_samples=5000)
X = clip_embeddings.numpy() if torch.is_tensor(clip_embeddings) else clip_embeddings
y = gender_labels.numpy() if torch.is_tensor(gender_labels) else gender_labels
clip_embs, attr_mat, idx_list, attr_names = get_all_embeddings_and_attrs(dataset)


bias_vectors = []
for attr in bias_attribute_names:
    idx = attr_names.index(attr)
    mu_pos = X[attr_mat[:, idx] == 1].mean(axis=0)
    mu_neg = X[attr_mat[:, idx] == 0].mean(axis=0)
    v = mu_pos - mu_neg
    v = v.astype(np.float32)
    v /= np.linalg.norm(v)
    bias_vectors.append(v)


B = np.stack(bias_vectors, axis=1)
Q, _ = np.linalg.qr(B)
bias_basis = Q.T                              #(2, D)

X_hard_yt = hard_debias(X, bias_basis)
X_soft_yt = soft_debias(X, bias_basis, lam=0)

clf, X_train, X_test, y_train, y_test, y_pred, misclassified_indices = train_gender_classifier(X_soft_yt, gender_labels)

gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)
eval_classifier(y_test, y_pred)

compare_groups_ttest(clf, clip_embeddings, attr_mat, attr_names, bias_attribute_names)

results = []

for attr in attr_names:
    # redirect print to capture
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        compare_groups_ttest(clf, clip_embeddings, attr_mat, attr_names, [attr])
        output = mystdout.getvalue()
        lines = output.strip().split("\n")

        t_line = [line for line in lines if "T-test result" in line]
        if t_line:
            t_str = t_line[0]
            t_val = float(t_str.split("t =")[1].split(",")[0].strip())
            p_val = float(t_str.split("p =")[1].strip())
            sig = "*" if p_val < 0.05 else ""
            output += f"Significant: {sig}\n"

        results.append(output)

    except Exception as e:
        print(f"Error on attribute {attr}: {e}")

    finally:
        sys.stdout = old_stdout

# === Print all results at once ===
print("\n" + "="*60)
print("T-TEST SUMMARY FOR ALL ATTRIBUTES (p < 0.05 marked with *)")
print("="*60)

for r in results:
    print("\n" + r)

"""
base:
Gender classifier training accuracy: 0.9958
Gender classifier test accuracy:     0.9881
Classification Report:
              precision    recall  f1-score   support

      Female       0.99      0.99      0.99       591
        Male       0.99      0.99      0.99       421

    accuracy                           0.99      1012
   macro avg       0.99      0.99      0.99      1012
weighted avg       0.99      0.99      0.99      1012

Confusion Matrix:
[[585   6]
 [  6 415]]

Average P(male) for Group 1 (female‐associated attributes)  = 0.9847  ± 0.0545
Average P(male) for Group 2 (none of those attributes)       = 0.9983  ± 0.0091

Two‐sample t‐test: t = -4.29,  p = 2.359e-05

hard:

Gender classifier training accuracy: 0.8687
Gender classifier test accuracy:     0.8063
Classification Report:
              precision    recall  f1-score   support

      Female       0.79      0.91      0.85       591
        Male       0.83      0.67      0.74       421

    accuracy                           0.81      1012
   macro avg       0.81      0.79      0.79      1012
weighted avg       0.81      0.81      0.80      1012

Confusion Matrix:
[[535  56]
 [140 281]]

Group 1 (male & no_beard & attractive & young):  308 examples
Group 2 (male & no no_beard & no attractive & no young):  241 examples

Average P(male) for Group 1 (female‐associated attributes)  = 0.9673  ± 0.0842
Average P(male) for Group 2 (none of those attributes)       = 0.3287  ± 0.2334

Two‐sample t‐test: t = 40.38,  p = 7.689e-121


soft lam 5:

Gender classifier training accuracy: 0.9941
Gender classifier test accuracy:     0.9901
Classification Report:
              precision    recall  f1-score   support

      Female       0.99      0.99      0.99       591
        Male       0.99      0.99      0.99       421

    accuracy                           0.99      1012
   macro avg       0.99      0.99      0.99      1012
weighted avg       0.99      0.99      0.99      1012

Confusion Matrix:
[[587   4]
 [  6 415]]

Group 1 (male & no_beard & attractive & young):  308 examples
Group 2 (male & no no_beard & no attractive & no young):  241 examples

Average P(male) for Group 1 (female‐associated attributes)  = 0.9824  ± 0.0490
Average P(male) for Group 2 (none of those attributes)       = 0.9726  ± 0.0325

Two‐sample t‐test: t = 2.81,  p = 5.209e-03

soft lam 2:

Gender classifier training accuracy: 0.9955
Gender classifier test accuracy:     0.9901
Classification Report:
              precision    recall  f1-score   support

      Female       0.99      0.99      0.99       591
        Male       0.99      0.99      0.99       421

    accuracy                           0.99      1012
   macro avg       0.99      0.99      0.99      1012
weighted avg       0.99      0.99      0.99      1012

Confusion Matrix:
[[586   5]
 [  5 416]]

Group 1 (male & no_beard & attractive & young):  308 examples
Group 2 (male & no no_beard & no attractive & no young):  241 examples

Average P(male) for Group 1 (female‐associated attributes)  = 0.9838  ± 0.0486
Average P(male) for Group 2 (none of those attributes)       = 0.9915  ± 0.0185

Two‐sample t‐test: t = -2.54,  p = 1.133e-02


soft lam 3.5:
Gender classifier training accuracy: 0.9946
Gender classifier test accuracy:     0.9891
Classification Report:
              precision    recall  f1-score   support

      Female       0.99      0.99      0.99       591
        Male       0.99      0.99      0.99       421

    accuracy                           0.99      1012
   macro avg       0.99      0.99      0.99      1012
weighted avg       0.99      0.99      0.99      1012

Confusion Matrix:
[[586   5]
 [  6 415]]

Group 1 (male & no_beard & attractive & young):  308 examples
Group 2 (male & no no_beard & no attractive & no young):  241 examples

Average P(male) for Group 1 (female‐associated attributes)  = 0.9831  ± 0.0484
Average P(male) for Group 2 (none of those attributes)       = 0.9831  ± 0.0255

Two‐sample t‐test: t = 0.01,  p = 9.936e-01
"""
