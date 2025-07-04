import os
import sys
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Prevent matplotlib from trying to display plots
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from adjustText import adjust_text
import seaborn as sns
from pathlib import Path

# Add the Janines path to import functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'janines_testcode'))

from clip_utils import load_celeba_dataset, get_labels, get_all_embeddings_and_attrs
from gender_classification import train_gender_classifier, eval_classifier, gender_classifier_accuracy
from embedding_analysis import plot_attribute_bias_directions
from attribute_bias_analysis import compare_male_groups_ttest
from debias import hard_debias, soft_debias

# Additional imports for customization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind


# ==================== CONFIGURATION ====================
# Feature Flags - Toggle each analysis component
TOGGLE_GENDER_CLASSIFICATION = True
TOGGLE_PCA_VISUALIZATION = False
TOGGLE_TSNE_VISUALIZATION = False
TOGGLE_ATTRIBUTE_BIAS_DIRECTIONS = False
TOGGLE_MISCLASSIFIED_VISUALIZATION = False 
TOGGLE_MALE_GROUP_COMPARISON = True
TOGGLE_DEBIASING_ANALYSIS = True
TOGGLE_INDIVIDUAL_ALPHA_OPTIMIZATION = True
TOGGLE_TTEST_PRINTS = False 
DISPLAY_NEGATIVE_CENTROIDS = True # Show negative centroids in attribute bias plots



# Dataset Configuration
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
DATASET_SPLIT = "train"
# MAX_SAMPLES = 100000
MAX_SAMPLES = 20000
BATCH_SIZE = 512

# Output Configuration
OUTPUT_DIR = "result_imgs_100k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualization Configuration
PLOT_STYLE = "darkgrid"  # seaborn style: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
FIGURE_DPI = 300
FONT_SIZE = 17
TITLE_SIZE = 20
LABEL_SIZE = 18

# PCA/t-SNE plot configuration
SCATTER_ALPHA = 0.6
SCATTER_SIZE = 50
COLOR_PALETTE = ["#FF6B6B", "#4ECDC4"]  # Red for Female, Teal for Male

# Attribute bias plot configuration
# ARROW_BIAS_DIRECTIONS = ["No_Beard", "Young", "Gray_Hair","Bald", "Pale_Skin"] # Attributes to highlight with arrows in the plots
# ATTRIBUTES_TO_SHOW = None  # None for all, or list like ["Male", "Young", "Attractive"]
ATTRIBUTES_TO_SHOW = ["Rosy_Cheeks","Attractive",  "Wearing_Lipstick",  "Gray_Hair","Bald","Pale_Skin","5_o_Clock_Shadow"] # Attributes to highlight with arrows in the plots
ARROW_BIAS_DIRECTIONS = ["Rosy_Cheeks","Attractive",  "Wearing_Lipstick",  "Gray_Hair","Bald","Pale_Skin", "5_o_Clock_Shadow"] # Attributes to highlight with arrows in the plots


BIAS_LINE_WIDTH = 4.0
BIAS_TEXT_SIZE = 10

# Debiasing Configuration
DEBIAS_LAMBDA = 3.5  # For soft debiasing
# BIAS_ATTRIBUTES = ["Wearing_Necklace", "Rosy_Cheeks", "Goatee", "Wearing_Lipstick",  "No_Beard"]  # Attributes to debias
# BIAS_ATTRIBUTES = ["No_Beard", "Young", "Gray_Hair","Bald", "Pale_Skin"] # Attributes to highlight with arrows in the plots
BIAS_ATTRIBUTES = ["5_o_Clock_Shadow", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Mustache", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]



# ==================== HELPER FUNCTIONS ====================

def setup_plot_style():
    """Configure matplotlib and seaborn styling"""
    sns.set_style(PLOT_STYLE)
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['figure.dpi'] = FIGURE_DPI


def save_plot(filename):
    """Save current plot to result_imgs directory"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()
    print(f"Saved plot: {filepath}")


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
        if (DISPLAY_NEGATIVE_CENTROIDS and np.isnan(neg_c).any()) or np.isnan(pos_c).any():
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

    ax.legend(handles=legend_elements, loc='upper right', fontsize=LABEL_SIZE, bbox_to_anchor=(1.25, 1))
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_plot("attribute_bias_all_with_arrows.png")
    
    # ===== PLOT 2: Only selected attributes for clarity =====
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    texts2 = []
    
    # Plot only selected attributes
    for name, color in color_map.items():
        a_idx = attr_names.index(name)
        if DISPLAY_NEGATIVE_CENTROIDS:
            neg_c = neg_centroids[a_idx]
        else:
            neg_c = np.nan
        pos_c = pos_centroids[a_idx]
        
        if (DISPLAY_NEGATIVE_CENTROIDS and np.isnan(neg_c).any()) or np.isnan(pos_c).any():
            continue
        
        # Plot centroids with distinct markers
        #
        if DISPLAY_NEGATIVE_CENTROIDS:
            ax.scatter(neg_c[0], neg_c[1], c=[color], s=300, alpha=0.9, 
              marker='o', edgecolors='black', linewidth=2.5)
        ax.scatter(pos_c[0], pos_c[1], c=[color], s=300, alpha=0.9,
                  marker='.', edgecolors='black', linewidth=2.5)
        
        # Draw arrow
        if DISPLAY_NEGATIVE_CENTROIDS:
            # Arrow from negative to positive centroid
            ax.annotate('', xy=pos_c, xytext=neg_c,
                       arrowprops=dict(arrowstyle='<|-|>', color=color, lw=BIAS_LINE_WIDTH, alpha=0.9, mutation_scale=25))
        else:
            # Arrow from center to positive centroid
            center = np.array([0.0, 0.0])
            ax.annotate('', xy=pos_c, xytext=center,
                       arrowprops=dict(arrowstyle='->', color=color, lw=BIAS_LINE_WIDTH, alpha=0.9, mutation_scale=25))
        
        # Add labels near the points
        if DISPLAY_NEGATIVE_CENTROIDS:
            direction = pos_c - neg_c
            direction_norm = direction / np.linalg.norm(direction)
            
            text_neg = ax.text(neg_c[0] - direction_norm[0]*0.2, 
                   neg_c[1] - direction_norm[1]*0.2, 
                   f'{name} (-)', 
                   fontsize=FONT_SIZE, fontweight='normal', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                            edgecolor=color, linewidth=1.0, alpha=0.9))
            texts2.append(text_neg)
            
            text_pos = ax.text(pos_c[0] + direction_norm[0]*0.2, 
                   pos_c[1] + direction_norm[1]*0.2, 
                   f'{name} (+)', 
                   fontsize=FONT_SIZE, fontweight='normal', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                            edgecolor=color, linewidth=1.0, alpha=0.9))
            texts2.append(text_pos)
        else:
            # Only label the positive centroid
            text_obj = ax.text(pos_c[0], pos_c[1], name, 
                   fontsize=FONT_SIZE, fontweight='normal', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                            edgecolor=color, linewidth=1.0, alpha=0.9))
            texts2.append(text_obj)
            
       
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

        if selected_points.size > 0:
            x_min, x_max = selected_points[:, 0].min(), selected_points[:, 0].max()
            y_min, y_max = selected_points[:, 1].min(), selected_points[:, 1].max()

            x_padding = (x_max - x_min) * 0.2
            y_padding = (y_max - y_min) * 0.2

            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add zero lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title(f"Bias Directions for: {', '.join(ARROW_BIAS_DIRECTIONS)}", 
                fontsize=TITLE_SIZE,wrap=True)
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)",fontsize=LABEL_SIZE)
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)",fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    adjust_text(texts2, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    plt.tight_layout()
    save_plot("attribute_bias_selected_only.png")
    
    # Call the original function for the console output
    print("\nAttribute bias analysis (console output):")
    result = plot_attribute_bias_directions(clip_embeddings, attr_matrix, attr_names)
    
    # Move the original plot if it exists
    if os.path.exists("attribute_bias_directions.png"):
        dest_path = os.path.join(OUTPUT_DIR, "attribute_bias_directions_original.png")
        shutil.move("attribute_bias_directions.png", dest_path)
        print(f"Also saved original visualization to {OUTPUT_DIR}/attribute_bias_directions_original.png")
    
    return result


def show_misclassified_custom(dataset, y_test, y_pred, misclassified_indices, n=5):
    """Visualize misclassified images from gender classification"""
    if len(misclassified_indices) == 0:
        print("No misclassified images to show!")
        return
    
    n = min(n, len(misclassified_indices))
    plt.figure(figsize=(2.5*n, 3))
    
    for i in range(n):
        idx = misclassified_indices[i]
        plt.subplot(1, n, i + 1)
        
        # Load image on demand
        img, _ = dataset[idx] # dataset returns (image, attributes)
        
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

def compare_male_groups_custom(clf, clip_embeddings, attr_matrix, attr_names):
    """Compare classification confidence for male groups with different attributes"""
    # Call the updated t-test function
    compare_male_groups_ttest(
        clf,
        clip_embeddings,
        attr_matrix,
        attr_names,
        figsize=(10, 6),
        title_fontsize=TITLE_SIZE,
        label_fontsize=LABEL_SIZE, TOGGLE_TTEST_PRINTS=TOGGLE_TTEST_PRINTS
    )

    # Move the generated plots to the output directory
    attributes_to_test = [attr for attr in attr_names if attr != "Male"]
    for attribute in attributes_to_test:
        source_path = f"male_groups_confidence_hist_{attribute}.png"
        if os.path.exists(source_path):
            dest_path = os.path.join(OUTPUT_DIR, f"male_groups_confidence_hist_{attribute}.png")
            shutil.move(source_path, dest_path)
            print(f"Saved and moved plot: {dest_path}")

def find_optimal_alpha(attribute, X, attr_mat, attr_names, gender_labels, alphas=np.linspace(0.0, 1.0, 20)):
    """Find the optimal alpha for a single attribute that minimizes the t-statistic."""
    idx = attr_names.index(attribute)
    
    mask_pos = (attr_mat[:, idx] == 1)
    mask_neg = (attr_mat[:, idx] == 0)

    if np.sum(mask_pos) < 2 or np.sum(mask_neg) < 2:
        return None, float('inf')

    mu_pos = X[mask_pos].mean(axis=0)
    mu_neg = X[mask_neg].mean(axis=0)
    v = mu_pos - mu_neg
    v = v.astype(np.float32)
    
    if np.linalg.norm(v) < 1e-6:
        return None, float('inf')

    v /= np.linalg.norm(v)
    B = v.reshape(-1, 1)
    Q, _ = np.linalg.qr(B)
    bias_basis = Q.T

    best_alpha = None
    best_t = float("inf")

    for alpha in alphas:
    # for alpha in [0.78]:
        print(f"Testing alpha={alpha:.2f} for attribute '{attribute}'")
        X_soft = soft_debias(X, bias_basis, alpha=alpha)

        if np.isnan(X_soft).any():
            continue

        clf, _, _, _, _, _, _ = train_gender_classifier(X_soft, gender_labels)
        
        male_idx = attr_names.index("Male")
        male_mask = (attr_mat[:, male_idx] == 1)
        mask_group1 = male_mask & (attr_mat[:, idx] == 1)
        mask_group2 = male_mask & (attr_mat[:, idx] == 0)
        
        X_group1 = X_soft[mask_group1]
        X_group2 = X_soft[mask_group2]

        if len(X_group1) < 2 or len(X_group2) < 2:
            continue

        probs_group1 = clf.predict_proba(X_group1)[:, 1]
        probs_group2 = clf.predict_proba(X_group2)[:, 1]
        
        t_stat, _ = ttest_ind(probs_group1, probs_group2, equal_var=False)

        if abs(t_stat) < abs(best_t):
            best_t = t_stat
            best_alpha = alpha

    return best_alpha, best_t

def apply_individual_debiasing_alpha_qr(X, optimal_alphas, attr_mat, attr_names):
    """
    Apply individual debiasing using QR-decomposed orthogonal basis,
    consistent with the soft debiasing approach.
    """
    # Collect bias vectors for attributes with optimal alphas
    bias_directions = []
    alpha_values = []
    
    for attribute, alpha in optimal_alphas.items():
        if alpha is None or alpha == 0.0:
            continue
            
        idx = attr_names.index(attribute)
        mu_pos = X[attr_mat[:, idx] == 1].mean(axis=0)
        mu_neg = X[attr_mat[:, idx] == 0].mean(axis=0)
        v = mu_pos - mu_neg
        v = v.astype(np.float32)
        
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        v /= norm
        
        bias_directions.append(v)
        alpha_values.append(alpha)
    
    if not bias_directions:
        return X
    
    # Apply QR decomposition (same as soft debiasing)
    B = np.stack(bias_directions, axis=1)
    Q, _ = np.linalg.qr(B)
    bias_basis = Q.T
    
    # Use average alpha with orthogonal basis
    avg_alpha = np.mean(alpha_values)
    
    # Apply debiasing using orthogonal basis (same as soft_debias)
    return soft_debias(X, bias_basis, alpha=avg_alpha)

def apply_individual_debiasing_alpha_non_qr(X, optimal_alphas, attr_mat, attr_names):
    """
    Apply individual debiasing by summing non-orthogonal rank-1 projections in parallel.
    """
    total_projection = np.zeros_like(X)
    for attribute, alpha in optimal_alphas.items():
        if alpha is None or alpha == 0.0:
            continue
            
        idx = attr_names.index(attribute)
        mu_pos = X[attr_mat[:, idx] == 1].mean(axis=0)
        mu_neg = X[attr_mat[:, idx] == 0].mean(axis=0)
        v = mu_pos - mu_neg
        v = v.astype(np.float32)
        
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        v /= norm
        
        # Calculate individual projection and add to total
        total_projection += alpha * (X @ v.reshape(-1, 1)) * v

    return X - total_projection

def apply_soft_debias_with_qr_combined_basis(X, attributes_to_combine, attr_mat, attr_names, alpha):
    bias_directions = []
    for attr in attributes_to_combine:
        idx = attr_names.index(attr)
        mu_pos = X[attr_mat[:, idx] == 1].mean(axis=0)
        mu_neg = X[attr_mat[:, idx] == 0].mean(axis=0)
        v = mu_pos - mu_neg
        v = v.astype(np.float32)
        if np.linalg.norm(v) > 1e-6:
            v /= np.linalg.norm(v)
            bias_directions.append(v)
    
    if not bias_directions:
        return X.copy() # No bias directions to combine

    B = np.stack(bias_directions, axis=1)
    Q, _ = np.linalg.qr(B)
    bias_basis = Q.T
    
    return soft_debias(X, bias_basis, alpha=alpha)


def perform_debiasing_analysis(clip_embeddings, attr_matrix, attr_names, gender_labels):
    """Analyze the effect of debiasing on gender classification"""
    X = clip_embeddings.numpy() if torch.is_tensor(clip_embeddings) else clip_embeddings
    
    # --- Global Debiasing Setup ---
    bias_indices = [attr_names.index(attr) for attr in BIAS_ATTRIBUTES if attr in attr_names]
    bias_directions = []
    for idx in bias_indices:
        mu_pos = X[attr_matrix[:, idx] == 1].mean(axis=0)
        mu_neg = X[attr_matrix[:, idx] == 0].mean(axis=0)
        v = mu_pos - mu_neg
        v = v.astype(np.float32)
        if np.linalg.norm(v) > 1e-6:
            v /= np.linalg.norm(v)
            bias_directions.append(v)
    
    B = np.stack(bias_directions, axis=1)
    Q, _ = np.linalg.qr(B)
    bias_basis = Q.T
    
    X_hard = hard_debias(X, bias_basis)
    # For global soft debias, we still need to calculate an alpha from the DEBIAS_LAMBDA
    global_alpha = DEBIAS_LAMBDA / (1.0 + DEBIAS_LAMBDA)
    X_soft = soft_debias(X, bias_basis, alpha=global_alpha)

    # New: Soft Debias using QR Combined Basis (for comparison)
    X_soft_qr_combined = apply_soft_debias_with_qr_combined_basis(X, BIAS_ATTRIBUTES, attr_matrix, attr_names, global_alpha)
    
    # --- Analysis Setup ---
    print("\n" + "="*60)
    print("DEBIASING ANALYSIS")
    print("="*60)
    
    results = {}
    accuracies = {}
    all_ttest_results = {}
    optimal_alphas = {} # Initialize here to be in scope for the summary table
    
    # --- Initial T-Test to Find Biased Attributes ---
    clf_orig, _, _, _, _, _, _ = train_gender_classifier(X, gender_labels)
    initial_ttest_results = compare_male_groups_ttest(clf_orig, X, attr_matrix, attr_names, title="INITIAL T-TEST ANALYSIS")
    all_ttest_results["Original"] = initial_ttest_results
    
    # Bonferroni correction for significance
    significant_attributes = [attr for attr, res in initial_ttest_results.items() if res['p_val'] < (0.05 / len(initial_ttest_results))]
    
    # --- Individual Debiasing (Optional) ---
    X_individual = None
    if TOGGLE_INDIVIDUAL_ALPHA_OPTIMIZATION:
        print("\n" + "="*60)
        print("INDIVIDUAL DEBIASING OPTIMIZATION")
        print(f"Found {len(significant_attributes)} significantly biased attributes for optimization.")
        print("="*60)
        
        alpha_candidates = np.linspace(0.0, 1.0, 20)

        for attr in significant_attributes:
            original_t_stat = initial_ttest_results[attr]['t_stat']
            alpha, best_t = find_optimal_alpha(attr, X, attr_matrix, attr_names, gender_labels, alphas=alpha_candidates)
            
            if alpha is not None and abs(best_t) < abs(original_t_stat):
                optimal_alphas[attr] = alpha
            else:
                optimal_alphas[attr] = None # Mark as None if no improvement

        X_individual_qr = apply_individual_debiasing_alpha_qr(X, optimal_alphas, attr_matrix, attr_names)
        X_individual_non_qr = apply_individual_debiasing_alpha_non_qr(X, optimal_alphas, attr_matrix, attr_names)

    # --- Evaluate All Embedding Versions ---
    embedding_versions = [
        ("Original", X), 
        ("Hard Debias", X_hard), 
        (f"Soft Debias (α={global_alpha:.2f})", X_soft),
        (f"Soft Debias (QR Combined, α={global_alpha:.2f})", X_soft_qr_combined)
    ]
    if TOGGLE_INDIVIDUAL_ALPHA_OPTIMIZATION:
        embedding_versions.append(("IndivT-s", X_individual_non_qr))
        embedding_versions.append(("IndivQRT-s", X_individual_qr))

    for name, X_version in embedding_versions:
        print(f"\nAnalyzing {name} Embeddings:")
        clf, X_train, X_test, y_train, y_test, y_pred, _ = train_gender_classifier(X_version, gender_labels)
        acc = gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)
        results[name] = (clf, X_version)
        accuracies[name] = acc

        if name == "Individual Debias":
            print(f"    [Diagnostic] Gender classification accuracy on Individual Debias embeddings: {acc:.2%}")

        ttest_results = compare_male_groups_ttest(clf, X_version, attr_matrix, attr_names, 
                                    title=f"T-TEST - {name}", 
                                    output_prefix=f"{name.replace(' ', '_').lower()}_")
        all_ttest_results[name] = ttest_results

    # --- Visualization ---
    num_plots = len(results)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 8), sharey=True)
    if num_plots == 1: axes = [axes] # Ensure axes is always iterable

    all_X_2d = [PCA(n_components=2).fit_transform(res[1][:1000]) for res in results.values()]
    max_range = np.max(np.abs(np.vstack(all_X_2d))) * 1.1
    
    for i, (name, (clf, X_version)) in enumerate(results.items()):
        ax = axes[i]
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_version[:1000])
        
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=gender_labels[:1000].numpy(),
                   cmap=plt.cm.colors.ListedColormap(COLOR_PALETTE),
                   alpha=0.6, s=40, edgecolors='none')
        
        ax.set_title(f"{name}\nAccuracy: {accuracies[name]:.2%}", fontsize=TITLE_SIZE)
        ax.set_xlabel("PC 1")
        if i == 0: ax.set_ylabel("PC 2")
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.grid(True, alpha=0.4)
        ax.set_aspect('equal', 'box')

    legend_elements = [Patch(facecolor=COLOR_PALETTE[0], label='Female'), Patch(facecolor=COLOR_PALETTE[1], label='Male')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=LABEL_SIZE)
    plt.suptitle("Effect of Debiasing on Embedding Space", fontsize=TITLE_SIZE + 4, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_plot("debiasing_comparison.png")

    # --- T-Test Summary Table ---
    print("\n" + "="*132)
    print("DEBIASING T-TEST SUMMARY")
    print("="*132)
    
    header_parts = ["Attribute", "Orig", "Hard", f"Softα={global_alpha:.2f}", "Opti α", "IndivT-s", "IndivQRT-s", "∆Orig", "∆Soft", "∆SoftQR"]
    col_widths = [20, 7, 7, 12, 8, 11, 11, 6, 6, 8]

    header = " | ".join([f"{h:<{w}}" for h, w in zip(header_parts, col_widths)])
    print(header)
    print("-" * len(header))

    sorted_attributes = sorted(all_ttest_results["Original"].keys())
    soft_t_key = f"Soft Debias (α={global_alpha:.2f})"
    soft_qr_t_key = f"Soft Debias (QR Combined, α={global_alpha:.2f})"
    for attribute in sorted_attributes:
        original_t = all_ttest_results["Original"][attribute]['t_stat']
        hard_t = all_ttest_results["Hard Debias"][attribute]['t_stat']
        soft_t = all_ttest_results[soft_t_key][attribute]['t_stat']
        soft_qr_t = all_ttest_results[soft_qr_t_key][attribute]['t_stat']
        
        row_parts = [
            attribute,
            f"{original_t:.3f}",
            f"{hard_t:.3f}",
            f"{soft_t:.3f}"
        ]
        
        optimal_alpha_val = optimal_alphas.get(attribute)
        if optimal_alpha_val is not None:
            individual_non_qr_t = all_ttest_results["IndivT-s"][attribute]['t_stat']
            individual_qr_t = all_ttest_results["IndivQRT-s"][attribute]['t_stat']
            
            delta_orig = abs(individual_non_qr_t - original_t)
            delta_soft = abs(individual_non_qr_t - soft_t)
            delta_soft_qr = abs(individual_qr_t - soft_qr_t)
            
            row_parts.extend([
                f"{optimal_alpha_val:.2f}",
                f"{individual_non_qr_t:.3f}",
                f"{individual_qr_t:.3f}",
                f"{delta_orig:.3f}",
                f"{delta_soft:.3f}",
                f"{delta_soft_qr:.3f}"
            ])
        else:
            row_parts.extend(["-", "-", "-", "-", "-", "-"])

        row = " | ".join([f"{p:<{w}}" for p, w in zip(row_parts, col_widths)])
        print(row)
        
    return results




# ==================== MAIN PIPELINE ====================

def main():
    """Main pipeline orchestrating all bias analysis components"""
    setup_plot_style()
    print(f"Starting CLIP Bias Analysis Pipeline")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Load dataset - Core functionality needed by all components
    print("\nLoading CelebA dataset...")
    try:
        dataset = load_celeba_dataset(root=DATA_ROOT, split=DATASET_SPLIT, download=False)
        print(f"Successfully loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"ERROR loading dataset: {type(e).__name__}: {e}")
        print("\nPossible solutions:")
        print("1. Ensure CelebA dataset is properly downloaded in the data directory")
        print("2. Check that the data directory structure is correct")
        return
    
    # Get CLIP embeddings and labels - Extract features from images
    print(f"\nExtracting CLIP embeddings for {MAX_SAMPLES} samples...")
    clip_embeddings, gender_labels = get_labels(dataset, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES)
    
    # Get all embeddings with attributes - Full attribute matrix for bias analysis
    clip_embs_all, attr_mat, idx_list, attr_names = get_all_embeddings_and_attrs(
        dataset, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES
    )
    
    # 1. Gender Classification
    if TOGGLE_GENDER_CLASSIFICATION:
        print("\n" + "="*60)
        print("GENDER CLASSIFICATION ANALYSIS")
        print("="*60)
        # Train logistic regression classifier for gender prediction
        clf, X_train, X_test, y_train, y_test, y_pred, misclassified_indices = \
            train_gender_classifier(clip_embeddings, gender_labels)
        
        # Evaluate classifier performance
        gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)
        eval_classifier(y_test, y_pred)
    
    # 2. PCA Visualization
    if TOGGLE_PCA_VISUALIZATION:
        print("\nGenerating PCA visualization...")
        # Project embeddings to 2D using PCA and visualize gender separation
        pca_visualization(clip_embeddings, gender_labels)
    
    # 3. t-SNE Visualization
    if TOGGLE_TSNE_VISUALIZATION:
        print("\nGenerating t-SNE visualization...")
        # Non-linear dimensionality reduction for visualization
        tsne_visualization(clip_embeddings, gender_labels)
    
    # 4. Attribute Bias Directions
    if TOGGLE_ATTRIBUTE_BIAS_DIRECTIONS:
        print("\nAnalyzing attribute bias directions...")
        # Visualize how different attributes create directional biases in embedding space
        plot_attribute_bias_directions_custom(clip_embs_all, attr_mat, attr_names)
    
    # 5. Misclassified Visualization
    if TOGGLE_MISCLASSIFIED_VISUALIZATION and TOGGLE_GENDER_CLASSIFICATION:
        print("\nVisualizing misclassified examples...")
        # Show examples where gender classifier made mistakes
        show_misclassified_custom(dataset, y_test, y_pred, misclassified_indices, n=5)
    
    # 6. Male Group Comparison
    if TOGGLE_MALE_GROUP_COMPARISON and TOGGLE_GENDER_CLASSIFICATION:
        print("\nComparing male groups with different attributes...")
        # Statistical test comparing males with/without "female-associated" attributes
        compare_male_groups_custom(clf, clip_embeddings, attr_mat, attr_names)
    
    # 7. Debiasing Analysis
    if TOGGLE_DEBIASING_ANALYSIS:
        print("\nPerforming debiasing analysis...")
        # Apply and evaluate hard/soft debiasing techniques
        perform_debiasing_analysis(clip_embeddings, attr_mat, attr_names, gender_labels)
    
    print("\n" + "="*60)
    print(f"Analysis complete! All results saved to {OUTPUT_DIR}/")
    print("\nGenerated plots:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        if file.endswith('.png'):
            print(f"  - {file}")
    print("="*60)


if __name__ == "__main__":
    main()
