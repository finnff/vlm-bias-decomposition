import os
import sys
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Prevent matplotlib from trying to display plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from adjustText import adjust_text
import seaborn as sns
from pathlib import Path
import gc
import warnings
import csv
from scipy.stats import ttest_ind
from scipy.optimize import minimize_scalar
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")


# Add the Janines path to import functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'janines_testcode'))

from clip_utils import load_celeba_dataset, get_labels, get_all_embeddings_and_attrs
from gender_classification import train_gender_classifier, eval_classifier, gender_classifier_accuracy
from embedding_analysis import plot_attribute_bias_directions
from attribute_bias_analysis import compare_male_groups_ttest
# from debias import hard_debias, soft_debias # Using local versions now



# ==================== CONFIGURATION ====================
# Feature Flags - Toggle each analysis component
TOGGLE_GENDER_CLASSIFICATION = True
TOGGLE_PCA_VISUALIZATION = True
TOGGLE_TSNE_VISUALIZATION = True
TOGGLE_ATTRIBUTE_BIAS_DIRECTIONS = True
TOGGLE_MISCLASSIFIED_VISUALIZATION = True
TOGGLE_MALE_GROUP_COMPARISON = True
TOGGLE_DEBIASING_ANALYSIS = True
TOGGLE_INDIVIDUAL_ALPHA_CALCULATION = True # <<<< NEW TOGGLE
TOGGLE_DETAILED_MALE_GROUP_PLOTS = False # Toggle for creating plots for each attribute in male group comparison
DISPLAY_NEGATIVE_CENTROIDS = False # Show negative centroids in attribute bias plots



# Dataset Configuration
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
DATASET_SPLIT = "train"
MAX_SAMPLES = 100000 # Increased samples for more stable statistics
BATCH_SIZE = 256

# Output Configuration
OUTPUT_DIR = "result_imgs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GENERATED_PLOTS = [] # List to track generated plots for this run

# Visualization Configuration
PLOT_STYLE = "darkgrid"  # seaborn style: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
FIGURE_DPI = 300
FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20

# PCA/t-SNE plot configuration
SCATTER_ALPHA = 0.6
SCATTER_SIZE = 50
COLOR_PALETTE = ["#FF6B6B", "#4ECDC4"]  # Red for Female, Teal for Male

# Attribute bias plot configuration
ARROW_BIAS_DIRECTIONS = ["No_Beard", "Young", "Gray_Hair","Bald"] # Attributes to highlight with arrows in the plots
ATTRIBUTES_TO_SHOW = ["No_Beard", "Young", "Gray_Hair","Bald"]

BIAS_LINE_WIDTH = 1.5
BIAS_TEXT_SIZE = 10

# Debiasing Configuration
DEBIAS_LAMBDA = 3.5  # For soft debiasing (used if individual calculation is off)
BIAS_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
    "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face",
    "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
    "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
    "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]


# ==================== HELPER FUNCTIONS ====================

def setup_plot_style():
    """Configure matplotlib and seaborn styling"""
    sns.set_style(PLOT_STYLE)
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['figure.dpi'] = FIGURE_DPI


def save_plot(filename):
    """Save current plot to result_imgs directory and track it."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()
    print(f"Saved plot: {filepath}")
    if filename not in GENERATED_PLOTS:
        GENERATED_PLOTS.append(filename)

# ================= DEBIASING & OPTIMIZATION HELPERS =================

def hard_debias(X, bias_basis):
    """Projects out the bias basis from X."""
    P = bias_basis.T @ bias_basis
    return X - X @ P

def soft_debias(X, bias_basis, alpha):
    """Partially projects out the bias basis from X."""
    P = bias_basis.T @ bias_basis
    return X - alpha * (X @ P)

def train_gender_classifier_fast(X_tr, y_tr, X_te, y_te, max_iter=5):
    """A faster, less precise classifier for use in optimization."""
    clf = SGDClassifier(loss="log_loss", max_iter=max_iter, tol=1e-3, random_state=0)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    return clf, acc

def find_optimal_alpha(X, y, attr_mat, attr_names, attr_name, train_idx, test_idx):
    """
    Finds the optimal alpha for a single attribute to minimize the t-statistic.
    Adapted from janines_testcode/findLambda.py.
    """
    male_idx = attr_names.index("Male")
    mask_male = attr_mat[:, male_idx]
    y_tr_base, y_te_base = y[train_idx], y[test_idx]

    attr_idx = attr_names.index(attr_name)
    pos = attr_mat[:, attr_idx]
    neg = ~pos

    # Check for sufficient data
    if pos.sum() < 10 or neg.sum() < 10:
        # print(f"Skipping {attr_name} due to insufficient samples.")
        return None

    mu_pos = X[pos].mean(axis=0)
    mu_neg = X[neg].mean(axis=0)
    v = mu_pos - mu_neg
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        # print(f"Skipping {attr_name} due to degenerate bias vector.")
        return None

    v = (v / norm).astype(np.float32)
    P = np.outer(v, v)

    mask1_te = (mask_male & pos)[test_idx]
    mask2_te = (mask_male & neg)[test_idx]

    if mask1_te.sum() < 2 or mask2_te.sum() < 2:
        # print(f"Skipping {attr_name} due to insufficient test samples for t-test.")
        return None

    def t_stat_for_alpha(alpha):
        X_soft = X - alpha * (X @ P)
        X_tr, X_te = X_soft[train_idx], X_soft[test_idx]
        
        # Use the fast classifier for optimization
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
        bounds=(0.0, 1.0), # Search up to 1.0
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 50}, # Looser tolerance for speed
    )

    return res.x


# ==================== MODIFIED VISUALIZATION FUNCTIONS ====================

def pca_visualization(clip_embeddings, gender_labels):
    """Perform PCA and create 2D visualization of embeddings colored by gender"""
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(clip_embeddings.numpy())
    
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        X_reduced[:, 0], 
        X_reduced[:, 1], 
        c=gender_labels.numpy(),
        cmap=plt.cm.colors.ListedColormap(COLOR_PALETTE),
        alpha=SCATTER_ALPHA,
        s=SCATTER_SIZE,
        edgecolors='none'
    )
    
    plt.colorbar(scatter, ticks=[0, 1], label='Gender')
    plt.clim(-0.5, 1.5)
    
    plt.title("PCA of CLIP Embeddings Colored by Gender", fontsize=TITLE_SIZE)
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_PALETTE[0], label='Female'),
        Patch(facecolor=COLOR_PALETTE[1], label='Male')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("gender_pca_custom.png")

def tsne_visualization(clip_embeddings, gender_labels):
    """Perform t-SNE and create 2D visualization of embeddings colored by gender"""
    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(clip_embeddings.numpy())
    
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=gender_labels.numpy(),
        cmap=plt.cm.colors.ListedColormap(COLOR_PALETTE),
        alpha=SCATTER_ALPHA,
        s=SCATTER_SIZE,
        edgecolors='none'
    )
    
    plt.title("t-SNE of CLIP Embeddings Colored by Gender", fontsize=TITLE_SIZE)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_PALETTE[0], label='Female'),
        Patch(facecolor=COLOR_PALETTE[1], label='Male')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("gender_tsne_custom.png")

def plot_attribute_bias_directions_custom(clip_embeddings, attr_matrix, attr_names):
    """Visualize bias directions for attributes in PCA space with customization"""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(clip_embeddings.numpy())
    
    # Compute centroids for ALL attributes (no filtering)
    num_attrs = len(attr_names)
    if DISPLAY_NEGATIVE_CENTROIDS:
        neg_centroids = np.zeros((num_attrs, 2), dtype=float)
    pos_centroids = np.zeros((num_attrs, 2), dtype=float)
    
    for a_idx in range(num_attrs):
        mask_neg = (attr_matrix[:, a_idx] == 0)
        mask_pos = (attr_matrix[:, a_idx] == 1)
        
        if mask_neg.sum() < 10 or mask_pos.sum() < 10:
            if DISPLAY_NEGATIVE_CENTROIDS:
                neg_centroids[a_idx] = np.nan
            pos_centroids[a_idx] = np.nan
            continue
        
        if DISPLAY_NEGATIVE_CENTROIDS:
            neg_centroids[a_idx] = X_2d[mask_neg].mean(axis=0)
        pos_centroids[a_idx] = X_2d[mask_pos].mean(axis=0)
    
    # ===== PLOT 1: All attributes as dots, arrows only for selected =====
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    center = np.array([0.0, 0.0])

    # Define colors for selected attributes
    selected_colors = plt.cm.Set1(np.linspace(0, 0.8, len(ARROW_BIAS_DIRECTIONS)))
    color_map = {attr: color for attr, color in zip(ARROW_BIAS_DIRECTIONS, selected_colors)}

    texts = []   # Store labels for later adjustment

    # Plot all attributes
    for a_idx, name in enumerate(attr_names):
        if DISPLAY_NEGATIVE_CENTROIDS:
            neg_c = neg_centroids[a_idx]
        else:
            neg_c = np.nan
        pos_c = pos_centroids[a_idx]
        if np.isnan(neg_c).any() or np.isnan(pos_c).any():
            continue

        # styling
        is_sel = name in ARROW_BIAS_DIRECTIONS
        color = color_map.get(name, 'lightgray')
        size = 120 if is_sel else 60
        alpha = 0.9 if is_sel else 0.6
        lw = 2 if is_sel else 1

        # dots
        if DISPLAY_NEGATIVE_CENTROIDS:
            ax.scatter(*neg_c, c=[color], s=size, alpha=alpha,
                       marker='o', edgecolors='black', linewidth=lw)
        ax.scatter(*pos_c, c=[color], s=size, alpha=alpha,
                   marker='s', edgecolors='black', linewidth=lw)

        if is_sel:
            if DISPLAY_NEGATIVE_CENTROIDS:
                # arrow to negative centroid (dotted)
                ax.annotate(
                    '', xy=neg_c, xytext=center,
                    arrowprops=dict(arrowstyle='->', color=color, lw=BIAS_LINE_WIDTH,
                                    linestyle=':', alpha=0.8)
                )
            # arrow to positive centroid (solid)
            ax.annotate(
                '', xy=pos_c, xytext=center,
                arrowprops=dict(arrowstyle='->', color=color, lw=BIAS_LINE_WIDTH,
                                linestyle='-', alpha=0.8)
            )
            # label near the positive centroid
            txt = ax.text(
                pos_c[0], pos_c[1], name,
                fontsize=FONT_SIZE - 2, fontweight='normal',
                ha='center', va='center',
                bbox={ 'boxstyle':'round,pad=0.2',
                       'facecolor':'white',
                       'edgecolor': color,
                       'linewidth':0.8,
                       'alpha':0.9 }
            )
            texts.append(txt)

    
    # Set axis limits to ensure centering
    if DISPLAY_NEGATIVE_CENTROIDS:
        all_points = np.vstack([neg_centroids[~np.isnan(neg_centroids).any(axis=1)],
                               pos_centroids[~np.isnan(pos_centroids).any(axis=1)]])
    else:
        all_points = pos_centroids[~np.isnan(pos_centroids).any(axis=1)]
    max_range = np.max(np.abs(all_points)) * 1.1
    
    # Make plot square and centered
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    
    # Add zero lines for reference
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    
    ax.set_title("CelebA Attribute Bias Directions in 2D PCA Space", 
                fontsize=TITLE_SIZE)
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # Simplified legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', label='Negative Centroid (e.g., Not Attractive)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', label='Positive Centroid (e.g., Attractive)'),
        Patch(facecolor='lightgray', edgecolor='black', label='All Other Attributes'),
    ]
    # Add entries for highlighted attributes
    for name, color in color_map.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=name))

    if not DISPLAY_NEGATIVE_CENTROIDS:
        legend_elements.pop(0)

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, bbox_to_anchor=(1.25, 1))
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_plot("attribute_bias_all_with_arrows.png")
    
    # ===== PLOT 2: Only selected attributes for clarity =====
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Plot only selected attributes
    for name, color in color_map.items():
        a_idx = attr_names.index(name)
        if DISPLAY_NEGATIVE_CENTROIDS:
            neg_c = neg_centroids[a_idx]
        else:
            neg_c = np.nan
        pos_c = pos_centroids[a_idx]
        
        if np.isnan(neg_c).any() or np.isnan(pos_c).any():
            continue
        
        # Plot centroids with distinct markers
        if DISPLAY_NEGATIVE_CENTROIDS:
            ax.scatter(neg_c[0], neg_c[1], c=[color], s=300, alpha=0.9, 
                    marker='o', edgecolors='black', linewidth=2.5)
        ax.scatter(pos_c[0], pos_c[1], c=[color], s=300, alpha=0.9,
                marker='X', edgecolors='black', linewidth=2.5)
        
        # Draw arrow
        if DISPLAY_NEGATIVE_CENTROIDS:
            ax.annotate('', xy=pos_c, xytext=neg_c,
                    arrowprops=dict(arrowstyle='<|-|>', color=color, lw=3.5, alpha=0.9, mutation_scale=25))
        
        # Add labels near the points
        if DISPLAY_NEGATIVE_CENTROIDS:
            direction = pos_c - neg_c
            direction_norm = direction / np.linalg.norm(direction)
            
            ax.text(neg_c[0] - direction_norm[0]*0.2, 
                neg_c[1] - direction_norm[1]*0.2, 
                f'{name} (-)', 
                fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, linewidth=1.5, alpha=0.9))
        else:
            direction = pos_c
            direction_norm = direction / np.linalg.norm(direction)
        
        ax.text(pos_c[0] + direction_norm[0]*0.2, 
               pos_c[1] + direction_norm[1]*0.2, 
               f'{name} (+)', 
               fontsize=12, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color, linewidth=1.5, alpha=0.9))
    
    # Set centered axis limits for selected attributes only
    selected_indices = [attr_names.index(name) for name in ARROW_BIAS_DIRECTIONS 
                       if name in attr_names]
    if selected_indices:
        if DISPLAY_NEGATIVE_CENTROIDS:
            selected_points = np.vstack([
                neg_centroids[selected_indices][~np.isnan(neg_centroids[selected_indices]).any(axis=1)],
                pos_centroids[selected_indices][~np.isnan(pos_centroids[selected_indices]).any(axis=1)]
            ])
        else:
            selected_points = pos_centroids[selected_indices][~np.isnan(pos_centroids[selected_indices]).any(axis=1)]
        max_range = np.max(np.abs(selected_points)) * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
    
    # Add zero lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title(f"Bias Directions for: {', '.join(ARROW_BIAS_DIRECTIONS)}", 
                fontsize=TITLE_SIZE)
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    save_plot("attribute_bias_selected_only.png")
    
    # Call the original function for the console output
    print("\nAttribute bias analysis (console output):")
    plot_attribute_bias_directions(clip_embeddings, attr_matrix, attr_names)
    
    # Move the original plot if it exists
    if os.path.exists("attribute_bias_directions.png"):
        dest_path = os.path.join(OUTPUT_DIR, "attribute_bias_directions_original.png")
        shutil.move("attribute_bias_directions.png", dest_path)
        print(f"Also saved original visualization to {OUTPUT_DIR}/attribute_bias_directions_original.png")
        if "attribute_bias_directions_original.png" not in GENERATED_PLOTS:
            GENERATED_PLOTS.append("attribute_bias_directions_original.png")


def show_misclassified_custom(images, y_test, y_pred, misclassified_indices, n=5):
    """Visualize misclassified images from gender classification"""
    if len(misclassified_indices) == 0:
        print("No misclassified images to show!")
        return
    
    n = min(n, len(misclassified_indices))
    plt.figure(figsize=(2.5*n, 3))
    
    for i in range(n):
        idx = misclassified_indices[i]
        plt.subplot(1, n, i + 1)
        
        img = images[idx]
        # Denormalize CLIP preprocessing
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"True: {'Male' if y_test[idx] else 'Female'}\n"
                 f"Pred: {'Male' if y_pred[idx] else 'Female'}",
                 fontsize=FONT_SIZE)
        plt.axis('off')
    
    plt.suptitle(f"Misclassified Gender Images (Showing {n})", fontsize=TITLE_SIZE)
    plt.tight_layout()
    save_plot("misclassified_visualization.png")


def format_float(value, spec):
    """Helper to format floats or print 'n/a' for NaN values."""
    width = int(spec.split('.')[0])
    if np.isnan(value):
        return f"{'n/a':>{width}}"
    return f"{value:{spec}}"

def run_full_analysis(clip_embeddings, gender_labels, attr_matrix, attr_names, images):
    """Master function to run all analysis and debiasing steps."""
    
    # 1. Initial Gender Classification
    print("\n" + "="*60)
    print("GENDER CLASSIFICATION ANALYSIS (ORIGINAL EMBEDDINGS)")
    print("="*60)
    clf, X_train_orig, X_test_orig, y_train, y_test, y_pred, misclassified_indices = \
        train_gender_classifier(clip_embeddings, gender_labels)
    base_acc_overall = gender_classifier_accuracy(clf, X_train_orig, y_train, X_test_orig, y_test)
    eval_classifier(y_test, y_pred)

    # 2. Male Group Comparison for Original Embeddings
    if TOGGLE_MALE_GROUP_COMPARISON:
        print("\n--- Running Male Group T-Tests for Original Embeddings ---")
        _, new_plots = compare_male_groups_ttest(
            clf, clip_embeddings.numpy(), attr_matrix, attr_names,
            title="T-TESTS FOR ORIGINAL EMBEDDINGS",
            output_prefix="original_",
            generate_plots=TOGGLE_DETAILED_MALE_GROUP_PLOTS
        )
        GENERATED_PLOTS.extend(new_plots)

    # 3. Misclassified Visualization
    if TOGGLE_MISCLASSIFIED_VISUALIZATION:
        print("\nVisualizing misclassified examples...")
        show_misclassified_custom(images, y_test, y_pred, misclassified_indices, n=5)

    # 4. Debiasing Analysis
    if TOGGLE_DEBIASING_ANALYSIS:
        print("\n" + "="*140)
        print("DEBIASING ANALYSIS & INDIVIDUAL ALPHA OPTIMIZATION")
        print("="*140)
        
        X = clip_embeddings.numpy().astype(np.float32)
        y = gender_labels.numpy()
        attr_mat_bool = attr_matrix.astype(bool)
        
        # Pre-split data for all debiasing runs
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
        X_tr_base, X_te_base = X[train_idx], X[test_idx]
        y_tr_base, y_te_base = y[train_idx], y[test_idx]

        male_idx = attr_names.index("Male")
        mask_male = attr_mat_bool[:, male_idx]

        # --- Store results for averaging and CSV export ---
        results_data = []
        
        # --- Print Table Header ---
        header1 = (
            f"{'Attribute':<25} | {'Baseline':^31} | "
            f"{'Soft-Debias':^39} | {'Hard-Debias':^35}"
        )
        header2 = (
            f"{'':<25} | {'t':>8} | {'p':>11} | {'Acc%':>6} | "
            f"{'α':>5} | {'t':>8} | {'p':>11} | {'Acc%':>6} | "
            f"{'t':>8} | {'p':>11} | {'Acc%':>6}"
        )
        separator = (
            f"{'-'*25}-+-{'-'*8}-+-{'-'*11}-+-{'-'*6}-+-"
            f"{'-'*5}-+-{'-'*8}-+-{'-'*11}-+-{'-'*6}-+-"
            f"{'-'*8}-+-{'-'*11}-+-{'-'*6}"
        )
        print(header1)
        print(header2)
        print(separator)

        for attr_name in BIAS_ATTRIBUTES:
            if attr_name not in attr_names:
                # print(f"Warning: Attribute '{attr_name}' not found in dataset. Skipping.")
                continue

            attr_idx = attr_names.index(attr_name)
            pos = attr_mat_bool[:, attr_idx]
            neg = ~pos
            
            mask1 = mask_male & pos
            mask2 = mask_male & neg
            if mask1.sum() < 2 or mask2.sum() < 2:
                # print(f"Skipping {attr_name} due to insufficient samples in one group for t-test.")
                continue

            # --- Baseline Stats ---
            clf_base, base_acc = train_gender_classifier_fast(X_tr_base, y_tr_base, X_te_base, y_te_base, max_iter=10)
            base_acc *= 100
            base_probs = clf_base.predict_proba(X)[:, 1]
            base_t, base_p = ttest_ind(base_probs[mask1], base_probs[mask2], equal_var=False)

            # --- Initialize metrics and classifiers for this loop iteration ---
            alpha, soft_t, soft_p, soft_acc, hard_t, hard_p, hard_acc = [np.nan] * 7
            clf_soft, clf_hard = None, None
            
            # --- All debiasing happens in this block ---
            mu_pos = X[pos].mean(0)
            mu_neg = X[neg].mean(0)
            v = (mu_pos - mu_neg).astype(np.float32)
            
            if np.linalg.norm(v) > 1e-12:
                v /= (np.linalg.norm(v) + 1e-12)
                bias_basis = v.reshape(1, -1)

                # --- Alpha Calculation & Soft Debias ---
                optimal_alpha = np.nan
                if TOGGLE_INDIVIDUAL_ALPHA_CALCULATION:
                    res_alpha = find_optimal_alpha(X, y, attr_mat_bool, attr_names, attr_name, train_idx, test_idx)
                    if res_alpha is not None:
                        optimal_alpha = res_alpha
                else:
                    optimal_alpha = DEBIAS_LAMBDA
                
                alpha = optimal_alpha
                if not np.isnan(alpha):
                    X_soft = soft_debias(X, bias_basis, alpha)
                    X_tr_soft, X_te_soft = X_soft[train_idx], X_soft[test_idx]
                    clf_soft, soft_acc = train_gender_classifier_fast(X_tr_soft, y_tr_base, X_te_soft, y_te_base, max_iter=10)
                    soft_acc *= 100
                    soft_probs = clf_soft.predict_proba(X_soft)[:, 1]
                    soft_t, soft_p = ttest_ind(soft_probs[mask1], soft_probs[mask2], equal_var=False)

                # --- Hard Debias ---
                X_hard = hard_debias(X, bias_basis)
                X_tr_hard, X_te_hard = X_hard[train_idx], X_hard[test_idx]
                clf_hard, hard_acc = train_gender_classifier_fast(X_tr_hard, y_tr_base, X_te_hard, y_te_base, max_iter=10)
                hard_acc *= 100
                hard_probs = clf_hard.predict_proba(X_hard)[:, 1]
                hard_t, hard_p = ttest_ind(hard_probs[mask1], hard_probs[mask2], equal_var=False)

            # --- Store and Print Results Row ---
            row_data = {
                "Attribute": attr_name, "Baseline t": base_t, "Baseline p": base_p, "Baseline Acc": base_acc,
                "Alpha": alpha, "Soft-Debias t": soft_t, "Soft-Debias p": soft_p, "Soft-Debias Acc": soft_acc,
                "Hard-Debias t": hard_t, "Hard-Debias p": hard_p, "Hard-Debias Acc": hard_acc
            }
            results_data.append(row_data)
            
            print(
                f"{attr_name:<25} | "
                f"{format_float(base_t, '8.2f')} | {format_float(base_p, '11.3e')} | {format_float(base_acc, '6.1f')} | "
                f"{format_float(alpha, '5.3f')} | "
                f"{format_float(soft_t, '8.2f')} | {format_float(soft_p, '11.3e')} | {format_float(soft_acc, '6.1f')} | "
                f"{format_float(hard_t, '8.2f')} | {format_float(hard_p, '11.3e')} | {format_float(hard_acc, '6.1f')}"
            )
            
            # --- Clean up memory ---
            del clf_base
            if clf_soft: del clf_soft
            if clf_hard: del clf_hard
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Print Footer and Averages ---
        print(separator)
        
        avg_base_t = np.nanmean([np.abs(r['Baseline t']) for r in results_data])
        avg_base_p = np.nanmean([r['Baseline p'] for r in results_data])
        avg_base_acc = np.nanmean([r['Baseline Acc'] for r in results_data])
        avg_alpha = np.nanmean([r['Alpha'] for r in results_data])
        avg_soft_t = np.nanmean([np.abs(r['Soft-Debias t']) for r in results_data])
        avg_soft_p = np.nanmean([r['Soft-Debias p'] for r in results_data])
        avg_soft_acc = np.nanmean([r['Soft-Debias Acc'] for r in results_data])
        avg_hard_t = np.nanmean([np.abs(r['Hard-Debias t']) for r in results_data])
        avg_hard_p = np.nanmean([r['Hard-Debias p'] for r in results_data])
        avg_hard_acc = np.nanmean([r['Hard-Debias Acc'] for r in results_data])

        print(
            f"{'Averages (abs t)':<25} | "
            f"{format_float(avg_base_t, '8.2f')} | {format_float(avg_base_p, '11.3e')} | {format_float(avg_base_acc, '6.1f')} | "
            f"{format_float(avg_alpha, '5.3f')} | "
            f"{format_float(avg_soft_t, '8.2f')} | {format_float(avg_soft_p, '11.3e')} | {format_float(avg_soft_acc, '6.1f')} | "
            f"{format_float(avg_hard_t, '8.2f')} | {format_float(avg_hard_p, '11.3e')} | {format_float(avg_hard_acc, '6.1f')}"
        )
        print(separator)

        # --- Export to CSV ---
        csv_filepath = os.path.join(OUTPUT_DIR, 'debiasing_analysis_results.csv')
        if results_data:
            csv_header = list(results_data[0].keys())
            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writeheader()
                writer.writerows(results_data)
            print(f"\nTable exported to {csv_filepath}")


# ==================== MAIN PIPELINE ====================

def main():
    """Main pipeline orchestrating all bias analysis components"""
    setup_plot_style()
    print(f"Starting CLIP Bias Analysis Pipeline")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Load dataset
    print("\nLoading CelebA dataset...")
    try:
        dataset = load_celeba_dataset(root=DATA_ROOT, split=DATASET_SPLIT, download=False)
        print(f"Successfully loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"ERROR loading dataset: {type(e).__name__}: {e}")
        return
    
    # Get CLIP embeddings and labels
    print(f"\nExtracting CLIP embeddings for {MAX_SAMPLES} samples...")
    clip_embeddings, gender_labels, images = get_labels(dataset, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES)
    
    # Get all embeddings with attributes
    clip_embs_all, attr_matrix, _, attr_names = get_all_embeddings_and_attrs(
        dataset, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES
    )
    
    # Run core analysis functions based on toggles
    if TOGGLE_PCA_VISUALIZATION:
        print("\nGenerating PCA visualization...")
        pca_visualization(clip_embeddings, gender_labels)
    
    if TOGGLE_TSNE_VISUALIZATION:
        print("\nGenerating t-SNE visualization...")
        tsne_visualization(clip_embeddings, gender_labels)
    
    if TOGGLE_ATTRIBUTE_BIAS_DIRECTIONS:
        print("\nAnalyzing attribute bias directions...")
        plot_attribute_bias_directions_custom(clip_embs_all, attr_matrix, attr_names)
    
    # Run the main, consolidated analysis workflow
    if TOGGLE_GENDER_CLASSIFICATION:
        run_full_analysis(clip_embeddings, gender_labels, attr_matrix, attr_names, images)

    print("\n" + "="*60)
    print(f"Analysis complete! All results saved to {OUTPUT_DIR}/")
    print("\nGenerated plots in this run:")
    for file in sorted(GENERATED_PLOTS):
        print(f"  - {file}")
    print("="*60)


if __name__ == "__main__":
    main()
