import numpy as np
import torch
import gc
from scipy.stats import ttest_ind
from scipy.optimize import minimize_scalar
from clip_utils import get_labels, load_celeba_dataset, get_all_embeddings_and_attrs
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

# -- 1) Load and preprocess embeddings (100k samples)
dataset = load_celeba_dataset(root="./data", split="train", download=False)
clip_embeddings, gender_labels, _ = get_labels(dataset, batch_size=64, max_samples=100000)
X = clip_embeddings.cpu().numpy().astype(np.float32)
y = gender_labels.cpu().numpy()
_, attr_mat, _, attr_names = get_all_embeddings_and_attrs(dataset, max_samples=100000)
attr_mat = attr_mat.astype(bool)

male_idx = attr_names.index("Male")
mask_male = attr_mat[:, male_idx]

# -- 2) Pre-split indices for train/test
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)
y_tr_base, y_te_base = y[train_idx], y[test_idx]

# -- 3) Fast training function using SGDClassifier
def train_gender_classifier_fast(X_tr, y_tr, X_te, y_te, max_iter=10):
    clf = SGDClassifier(loss="log_loss", max_iter=max_iter, tol=1e-3, random_state=0)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    return clf, acc

# -- 4) Optimize alpha per attribute
results = []

for attr in attr_names:
    if attr == "Male":
        continue

    idx = attr_names.index(attr)
    pos = attr_mat[:, idx]
    neg = ~pos

    mu_pos = X[pos].mean(axis=0)
    mu_neg = X[neg].mean(axis=0)
    v = mu_pos - mu_neg
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        results.append((attr, None, None))
        print(f"{attr:25s} — skipped (degenerate vector)")
        continue

    v = (v / norm).astype(np.float32)
    P = np.outer(v, v)

    mask1 = mask_male & pos
    mask2 = mask_male & neg
    mask1_te = mask1[test_idx]
    mask2_te = mask2[test_idx]

    def t_stat_for_alpha(alpha):
        X_soft = X - alpha * (X @ P)
        X_tr, X_te = X_soft[train_idx], X_soft[test_idx]
        clf, _ = train_gender_classifier_fast(X_tr, y_tr_base, X_te, y_te_base)
        probs_test = clf.predict_proba(X_te)[:, 1]

        t, _ = ttest_ind(probs_test[mask1_te], probs_test[mask2_te], equal_var=False)
        del clf, probs_test, X_soft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return abs(t) if not np.isnan(t) else 1e6

    res = minimize_scalar(
        t_stat_for_alpha,
        bounds=(0.0, 0.99),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 100}
    )

    best_alpha = res.x

    # re-evaluate final t-stat at best alpha
    X_soft = X - best_alpha * (X @ P)
    probs_test_final = SGDClassifier(loss="log_loss", max_iter=5).fit(
        X_soft[train_idx], y_tr_base).predict_proba(X_soft[test_idx])[:, 1]
    t, _ = ttest_ind(probs_test_final[mask1_te], probs_test_final[mask2_te], equal_var=False)

    results.append((attr, best_alpha, abs(t)))
    print(f"{attr:25s}  α = {best_alpha:.3f}   |t| = {abs(t):.2f}")

