import numpy as np
import torch
from clip_utils import get_labels, load_celeba_dataset, get_all_embeddings_and_attrs
from gender_classification import train_gender_classifier, gender_classifier_accuracy, eval_classifier
from attribute_bias_analysis import compare_male_groups_ttest

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


idx_young      = attr_names.index("Young")
idx_attractive = attr_names.index("Attractive")
idx_beard      = attr_names.index("No_Beard")
temp           = attr_names.index("No_Beard")

mu_young_pos      = X[attr_mat[:, idx_young] == 1].mean(axis=0)   #u⁺
mu_young_neg      = X[attr_mat[:, idx_young] == 0].mean(axis=0)   #u⁻
mu_attr_pos       = X[attr_mat[:, idx_attractive] == 1].mean(axis=0)
mu_attr_neg       = X[attr_mat[:, idx_attractive] == 0].mean(axis=0)
mu_beard_pos      = X[attr_mat[:, idx_beard] == 1].mean(axis=0)   #u⁺
mu_beard_neg      = X[attr_mat[:, idx_beard] == 0].mean(axis=0)   #u⁻
mu_temp_pos      = X[attr_mat[:, temp] == 1].mean(axis=0)   #u⁺
mu_temp_neg      = X[attr_mat[:, temp] == 0].mean(axis=0)   #u⁻

v_young      = mu_young_pos - mu_young_neg
v_attractive = mu_attr_pos  - mu_attr_neg
v_beard = mu_beard_pos - mu_beard_neg
v_temp = mu_temp_pos - mu_temp_neg

#stupid pythonnn typinngggg
v_young      = v_young.astype(np.float32)
v_attractive = v_attractive.astype(np.float32)
v_beard = v_beard.astype(np.float32)
v_temp = v_temp.astype(np.float32)

v_young      /= np.linalg.norm(v_young)
v_attractive /= np.linalg.norm(v_attractive)
v_beard /= np.linalg.norm(v_beard)
v_temp /= np.linalg.norm(v_temp)

#B = np.stack([v_young, v_attractive, v_beard], axis=1)    #(D, 3) D = direction\
B = np.stack([v_beard,v_young,v_attractive], axis=1)
Q, _ = np.linalg.qr(B)                           #Q: (D, 3) orthonormal
bias_basis = Q.T                                 #(3, D)

X_hard_yt = hard_debias(X, bias_basis)
X_soft_yt = soft_debias(X, bias_basis, lam=3.5)

clf, X_train, X_test, y_train, y_test, y_pred, misclassified_indices = train_gender_classifier(X_soft_yt, gender_labels)

gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)
eval_classifier(y_test, y_pred)

compare_male_groups_ttest(clf, X_soft_yt, attr_mat, attr_names)


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
