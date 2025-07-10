import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def compare_male_groups_ttest(clf, clip_embeddings, attr_matrix, attr_names, title="T-TEST ANALYSIS FOR INDIVIDUAL MALE ATTRIBUTES", output_prefix="", figsize=(10, 6), title_fontsize=16, label_fontsize=14, TOGGLE_TTEST_PRINTS=True, generate_plots=True):
    """
    Performs and visualizes t-tests for all attributes within the male subgroup.

    For each attribute, this function compares the gender
    classifier's confidence (P(male)) between two groups of males:
    1. Males WITH the attribute.
    2. Males WITHOUT the attribute.

    It prints the t-statistic and p-value for each comparison, indicates significance
    with color, and optionally generates a comparative histogram.
    """
    male_idx = attr_names.index("Male")
    male_mask = (attr_matrix[:, male_idx] == 1)

    # Terminal colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[0m'

    print("\n" + "="*60)
    print(title)
    print("="*60)

    attributes_to_test = [attr for attr in attr_names if attr != "Male"]
    
    # Bonferroni correction
    num_tests = len(attributes_to_test)
    alpha = 0.05 / num_tests
    print(f"Applying Bonferroni correction. Significance level (alpha) = {alpha:.3e}")

    results = {}
    generated_plots = []

    for attribute in attributes_to_test:
        attr_idx = attr_names.index(attribute)

        # Group 1: Males WITH the attribute
        mask_group1 = male_mask & (attr_matrix[:, attr_idx] == 1)
        idxs_group1 = np.where(mask_group1)[0]

        # Group 2: Males WITHOUT the attribute
        mask_group2 = male_mask & (attr_matrix[:, attr_idx] == 0)
        idxs_group2 = np.where(mask_group2)[0]

        if TOGGLE_TTEST_PRINTS:
            if len(idxs_group1) < 10 or len(idxs_group2) < 10:
                print(f"\n--- Attribute: {attribute} ---")
                print(f"Skipping due to insufficient sample size: "
                      f"Group 1 ({attribute}) has {len(idxs_group1)} samples, "
                      f"Group 2 (Not {attribute}) has {len(idxs_group2)} samples.")
                continue
    
            print(f"\n--- Attribute: {attribute} ---")
            print(f"Group 1 (Males with {attribute}): {len(idxs_group1)} examples")
            print(f"Group 2 (Males without {attribute}): {len(idxs_group2)} examples")

        X_group1 = clip_embeddings[idxs_group1]
        X_group2 = clip_embeddings[idxs_group2]

        probs_group1 = clf.predict_proba(X_group1)[:, 1]
        probs_group2 = clf.predict_proba(X_group2)[:, 1]

        avg1, std1 = probs_group1.mean(), probs_group1.std()
        avg2, std2 = probs_group2.mean(), probs_group2.std()

        print(f"Avg. P(male) for Group 1 (with {attribute}): {avg1:.4f} \u00b1 {std1:.4f}")
        print(f"Avg. P(male) for Group 2 (without {attribute}): {avg2:.4f} \u00b1 {std2:.4f}")

        t_stat, p_val = ttest_ind(probs_group1, probs_group2, equal_var=False)
        
        results[attribute] = {'t_stat': t_stat, 'p_val': p_val}

        if p_val < alpha:
            print(GREEN + f"SIGNIFICANT (p < {alpha:.3e}): t = {t_stat:.2f}, p = {p_val:.3e}" + ENDC)
        else:
            print(RED + f"NOT SIGNIFICANT (p >= {alpha:.3e}): t = {t_stat:.2f}, p = {p_val:.3e}" + ENDC)

        # Visualization
        if generate_plots:
            plt.figure(figsize=figsize)
            sns.histplot(probs_group1, bins=50, alpha=0.7, label=f"With {attribute}", color="#4ECDC4", kde=True)
            sns.histplot(probs_group2, bins=50, alpha=0.7, label=f"Without {attribute}", color="#FF6B6B", kde=True)
            plt.legend(fontsize=label_fontsize)
            plt.title(f"P(male) Distribution for Males With vs. Without '{attribute}'", fontsize=title_fontsize)
            plt.xlabel("Predicted Probability of Being Male", fontsize=label_fontsize)
            plt.ylabel("Count", fontsize=label_fontsize)
            plt.xticks(fontsize=label_fontsize)
            plt.yticks(fontsize=label_fontsize)
            plt.tight_layout()
            
            plot_filename = f"{output_prefix}male_groups_confidence_hist_{attribute}.png"
            plt.savefig(f"result_imgs/{plot_filename}")
            generated_plots.append(plot_filename)
            # plt.show() # Disabled for non-interactive use
            plt.close()
    print("\n" + "="*60)
    return results, generated_plots



