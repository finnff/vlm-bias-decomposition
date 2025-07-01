import numpy as np
from sklearn.metrics import log_loss
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def compare_groups_ttest(clf, clip_embeddings, attr_matrix, attr_names, attributes_to_split_on, fixed_filter="Male"):
    """
    Group 1 has all of the specified attributes
    Group 2 has none of them
    """

    if fixed_filter:
        filter_idx = attr_names.index(fixed_filter)
        filter_mask = attr_matrix[:, filter_idx] == 1
    else:
        filter_mask = np.ones(len(attr_matrix), dtype=bool)  # no filtering

    # Get indices of the selected attributes
    attr_idxs = [attr_names.index(attr) for attr in attributes_to_split_on]

    # Group 1: has all specified attributes (== 1)
    mask_group1 = filter_mask.copy()
    for idx in attr_idxs:
        mask_group1 &= (attr_matrix[:, idx] == 1)
    idxs_group1 = np.where(mask_group1)[0]

    # Group 2: has none of the specified attributes (== 0)
    mask_group2 = filter_mask.copy()
    for idx in attr_idxs:
        mask_group2 &= (attr_matrix[:, idx] == 0)
    idxs_group2 = np.where(mask_group2)[0]

    print(f"\nSelected Attributes: {attributes_to_split_on}")
    print(f"Group 1 (has all):   {len(idxs_group1)} samples")
    print(f"Group 2 (has none):  {len(idxs_group2)} samples\n")

    # Run prediction
    X_group1 = clip_embeddings[idxs_group1]
    X_group2 = clip_embeddings[idxs_group2]

    probs_group1 = clf.predict_proba(X_group1)[:, 1]
    probs_group2 = clf.predict_proba(X_group2)[:, 1]

    avg1, std1 = probs_group1.mean(), probs_group1.std()
    avg2, std2 = probs_group2.mean(), probs_group2.std()

    print(f"Avg P(male) Group 1 (has all):   {avg1:.4f} ± {std1:.4f}")
    print(f"Avg P(male) Group 2 (has none): {avg2:.4f} ± {std2:.4f}")

    t_stat, p_val = ttest_ind(probs_group1, probs_group2, equal_var=False)
    print(f"T-test result: t = {t_stat:.2f}, p = {p_val:.3e}")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.hist(probs_group1, bins=50, alpha=0.6, label="Group 1 (has all)")
    plt.hist(probs_group2, bins=50, alpha=0.6, label="Group 2 (has none)")
    plt.legend()
    plt.title("Predicted P(male) Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("group_comparison_hist.png")
    plt.close()
