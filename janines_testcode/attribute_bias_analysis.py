import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def compare_male_groups_ttest(clf, clip_embeddings, attr_matrix, attr_names):
    male_idx = attr_names.index("Male")
    no_beard_idx = attr_names.index("No_Beard")
    attractive_idx = attr_names.index("Attractive")
    young_idx = attr_names.index("Young")

    # Group 1: male & no beard & attractive & young
    mask_group1 = (
        (attr_matrix[:, male_idx] == 1) &
        (attr_matrix[:, no_beard_idx] == 1) &
        (attr_matrix[:, attractive_idx] == 1) &
        (attr_matrix[:, young_idx] == 1)
    )
    idxs_group1 = np.where(mask_group1)[0]

    # Group 2: male & no no_beard & no attractive & no young
    mask_group2 = (
        (attr_matrix[:, male_idx] == 1) &
        (attr_matrix[:, no_beard_idx] == 0) &
        (attr_matrix[:, attractive_idx] == 0) &
        (attr_matrix[:, young_idx] == 0)
    )
    idxs_group2 = np.where(mask_group2)[0]

    print(f"\nGroup 1 (male & no_beard & attractive & young):  {len(idxs_group1)} examples")
    print(f"Group 2 (male & no no_beard & no attractive & no young):  {len(idxs_group2)} examples\n")

    X_group1 = clip_embeddings[idxs_group1] #.numpy()
    X_group2 = clip_embeddings[idxs_group2] #.numpy()

    probs_group1 = clf.predict_proba(X_group1)[:, 1]
    probs_group2 = clf.predict_proba(X_group2)[:, 1]

    avg1 = probs_group1.mean()
    std1 = probs_group1.std()
    avg2 = probs_group2.mean()
    std2 = probs_group2.std()

    print(f"Average P(male) for Group 1 (female‐associated attributes)  = {avg1:.4f}  ± {std1:.4f}")
    print(f"Average P(male) for Group 2 (none of those attributes)       = {avg2:.4f}  ± {std2:.4f}\n")

    t_stat, p_val = ttest_ind(probs_group1, probs_group2, equal_var=False)
    print(f"Two‐sample t‐test: t = {t_stat:.2f},  p = {p_val:.3e}")


    plt.figure(figsize=(8, 4))
    plt.hist(probs_group1, bins=50, alpha=0.6, label="Group 1 (female‐associated attributes)")
    plt.hist(probs_group2, bins=50, alpha=0.6, label="Group 2 (none of those attributes)")
    plt.legend()
    plt.title("Predicted P(male) Distribution for Male Groups")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("male_groups_confidence_hist_1.png")
    plt.show()
