import numpy as np
import torch, gc
from clip_utils import get_labels, load_celeba_dataset, get_all_embeddings_and_attrs
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io, sys
from scipy.stats import ttest_ind

# -- 1) Load embeddings & attrs
dataset = load_celeba_dataset(root="./data", split="train", download=False)
clip_embs, gender_labels, _ = get_labels(dataset, batch_size=64, max_samples=100000)
X = clip_embs.cpu().numpy().astype(np.float32)
y = gender_labels.cpu().numpy()
_, attr_mat, _, attr_names = get_all_embeddings_and_attrs(dataset, max_samples=100000)
attr_mat = attr_mat.astype(bool)

male_idx = attr_names.index("Male")
mask_male = attr_mat[:, male_idx]

# -- 2) Pre-split once
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_tr_base, X_te_base = X[train_idx], X[test_idx]
y_tr_base, y_te_base = y[train_idx], y[test_idx]

# -- 3) Fast sklearn SGD trainer
def train_gender_classifier_fast(X_tr, y_tr, X_te, y_te, max_iter=5):
    clf = SGDClassifier(loss="log_loss", max_iter=max_iter, tol=1e-3, random_state=0)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    return clf, acc


#computing these takes ~20 min for me
best_alphas = {
    "5_o_Clock_Shadow":    0.903,
    "Arched_Eyebrows":     0.874,
    "Attractive":          0.020,
    "Bags_Under_Eyes":     0.895,
    "Bald":                0.810,
    "Bangs":               0.989,
    "Big_Lips":            0.984,
    "Big_Nose":            0.874,
    "Black_Hair":          0.231,
    "Blond_Hair":          0.990,
    "Blurry":              0.025,
    "Brown_Hair":          0.139,
    "Bushy_Eyebrows":      0.879,
    "Chubby":              0.816,
    "Double_Chin":         0.857,
    "Eyeglasses":          0.037,
    "Goatee":              0.854,
    "Gray_Hair":           0.820,
    "Heavy_Makeup":        0.234,
    "High_Cheekbones":     0.003,
    "Mouth_Slightly_Open": 0.743,
    "Mustache":            0.845,
    "Narrow_Eyes":         0.963,
    "No_Beard":            0.900,
    "Oval_Face":           0.047,
    "Pale_Skin":           0.946,
    "Pointy_Nose":         0.042,
    "Receding_Hairline":   0.492,
    "Rosy_Cheeks":         0.144,
    "Sideburns":           0.847,
    "Smiling":             0.049,
    "Straight_Hair":       0.988,
    "Wavy_Hair":           0.874,
    "Wearing_Earrings":    0.266,
    "Wearing_Hat":         0.051,
    "Wearing_Lipstick":    0.235,
    "Wearing_Necklace":    0.378,
    "Wearing_Necktie":     0.879,
    "Young":               0.844
}



# -- 5) Hard & soft debias helper
def hard_debias(X, bias_basis):
    P = bias_basis.T @ bias_basis  # (D, D)
    return X - X @ P

def soft_debias(X, bias_basis, alpha):
    P = bias_basis.T @ bias_basis
    return X - alpha * (X @ P)

# -- 6) Baseline: report base test acc & t-test
print("Baseline classifier on original embeddings:")
clf_base, base_acc = train_gender_classifier_fast(X_tr_base, y_tr_base, X_te_base, y_te_base, max_iter=10)
print(f"  Test Accuracy: {base_acc:.4f}")

print("\nBaseline bias t-test for each attribute:")
for attr in best_alphas:
    idx = attr_names.index(attr)
    mask1 = mask_male & attr_mat[:, idx]
    mask2 = mask_male & ~attr_mat[:, idx]

    probs = clf_base.predict_proba(X)[:,1]
    t, p = ttest_ind(probs[mask1], probs[mask2], equal_var=False)
    print(f"{attr:25s}  t = {t:.2f}, p = {p:.3e}")

# -- 7) Soft debias results
print("\nSoft‐Debiasing Results:")
soft_results = []
for attr, alpha in best_alphas.items():
    idx = attr_names.index(attr)
    pos = attr_mat[:, idx]
    mu_pos = X[pos].mean(0)
    mu_neg = X[~pos].mean(0)
    v = (mu_pos - mu_neg).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    basis = v.reshape(1, -1)

    X_soft = soft_debias(X, basis, alpha)
    # split
    X_tr, X_te = X_soft[train_idx], X_soft[test_idx]

    clf, acc = train_gender_classifier_fast(X_tr, y_tr_base, X_te, y_te_base, max_iter=5)
    probs = clf.predict_proba(X_soft)[:,1]
    t, p = ttest_ind(probs[mask_male & pos], probs[mask_male & ~pos], equal_var=False)

    soft_results.append((attr, alpha, acc, t, p))
    print(f"{attr:25s} α={alpha:.3f}  Acc={acc:.3f}  t={t:.2f}  p={p:.3e}")

    del clf, X_soft, probs
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# -- 8) Hard debias results
print("\nHard‐Debiasing Results:")
hard_results = []
for attr, alpha in best_alphas.items():
    idx = attr_names.index(attr)
    pos = attr_mat[:, idx]
    mu_pos = X[pos].mean(0)
    mu_neg = X[~pos].mean(0)
    v = (mu_pos - mu_neg).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    basis = v.reshape(1, -1)

    X_hard = hard_debias(X, basis)
    X_tr, X_te = X_hard[train_idx], X_hard[test_idx]

    clf, acc = train_gender_classifier_fast(X_tr, y_tr_base, X_te, y_te_base, max_iter=5)
    probs = clf.predict_proba(X_hard)[:,1]
    t, p = ttest_ind(probs[mask_male & pos], probs[mask_male & ~pos], equal_var=False)

    hard_results.append((attr, acc, t, p))
    print(f"{attr:25s} Acc={acc:.3f}  t={t:.2f}  p={p:.3e}")

    del clf, X_hard, probs
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
