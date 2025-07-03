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


# ==================== CONFIGURATION ====================
# Feature Flags - Toggle each analysis component
TOGGLE_GENDER_CLASSIFICATION = True
TOGGLE_PCA_VISUALIZATION = True 
TOGGLE_TSNE_VISUALIZATION = True
TOGGLE_ATTRIBUTE_BIAS_DIRECTIONS = True
TOGGLE_MISCLASSIFIED_VISUALIZATION = False 
TOGGLE_MALE_GROUP_COMPARISON = True 
TOGGLE_DEBIASING_ANALYSIS = True
DISPLAY_NEGATIVE_CENTROIDS = False # Show negative centroids in attribute bias plots



# Dataset Configuration
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
DATASET_SPLIT = "train"
# MAX_SAMPLES = 100000
MAX_SAMPLES = 100000
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
BIAS_ATTRIBUTES = ["Rosy_Cheeks","Attractive",  "Wearing_Lipstick",  "Gray_Hair","Bald","Pale_Skin", "5_o_Clock_Shadow"] # Attributes to highlight with arrows in the plots



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
        label_fontsize=LABEL_SIZE
    )

    # Move the generated plots to the output directory
    attributes_to_test = [attr for attr in attr_names if attr != "Male"]
    for attribute in attributes_to_test:
        source_path = f"male_groups_confidence_hist_{attribute}.png"
        if os.path.exists(source_path):
            dest_path = os.path.join(OUTPUT_DIR, f"male_groups_confidence_hist_{attribute}.png")
            shutil.move(source_path, dest_path)
            print(f"Saved and moved plot: {dest_path}")

def perform_debiasing_analysis(clip_embeddings, attr_matrix, attr_names, gender_labels):
    """Analyze the effect of debiasing on gender classification"""
    X = clip_embeddings.numpy() if torch.is_tensor(clip_embeddings) else clip_embeddings
    
    # Find bias attribute indices
    bias_indices = [attr_names.index(attr) for attr in BIAS_ATTRIBUTES]
    
    # Compute bias directions
    bias_directions = []
    for idx in bias_indices:
        mu_pos = X[attr_matrix[:, idx] == 1].mean(axis=0)
        mu_neg = X[attr_matrix[:, idx] == 0].mean(axis=0)
        v = mu_pos - mu_neg
        v = v.astype(np.float32)
        v /= np.linalg.norm(v)
        bias_directions.append(v)
    
    # Create bias basis
    B = np.stack(bias_directions, axis=1)
    Q, _ = np.linalg.qr(B)
    bias_basis = Q.T
    
    # Apply debiasing
    X_hard = hard_debias(X, bias_basis)
    X_soft = soft_debias(X, bias_basis, lam=DEBIAS_LAMBDA)
    
    # Train classifiers on each version
    print("\n" + "="*60)
    print("DEBIASING ANALYSIS")
    print("="*60)
    
    results = {}
    accuracies = {}
    all_ttest_results = {}

    for name, X_version in [("Original", X), ("Hard Debias", X_hard), 
                            ("Soft Debias(" + "\u03bb" + "={})".format(DEBIAS_LAMBDA), X_soft)]:
        print(f"\n{name} Embeddings:")
        clf, X_train, X_test, y_train, y_test, y_pred, _ = train_gender_classifier(X_version, gender_labels)
        acc = gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)
        results[name] = (clf, X_version)
        accuracies[name] = acc

        # Run t-test analysis for each debiasing method
        ttest_results = compare_male_groups_ttest(clf, X_version, attr_matrix, attr_names, 
                                    title=f"T-TEST ANALYSIS - {name}", 
                                    output_prefix=f"{name.replace(' ', '_').lower()}_")
        all_ttest_results[name] = ttest_results

    # Visualize debiasing effect
    fig, axes = plt.subplots(1, 3, figsize=(18, 8) )
    
    # Calculate global axis limits for consistent scaling
    all_X_2d = []
    for name, (clf, X_version) in results.items():
        pca_temp = PCA(n_components=2)
        X_2d_temp = pca_temp.fit_transform(X_version[:1000])
        all_X_2d.append(X_2d_temp)
    
    # Find the maximum range across all plots for consistent scaling
    all_points = np.vstack(all_X_2d)
    max_range = np.max(np.abs(all_points)) * 1.1
    
    for i, (name, (clf, X_version)) in enumerate(results.items()):
        ax = axes[i]
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_version[:1000])  # Subsample for clarity
        
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                            c=gender_labels[:1000].numpy(),
                            cmap=plt.cm.colors.ListedColormap(COLOR_PALETTE),
                            alpha=0.6, s=40, edgecolors='none')
        
        ax.set_title(f"{name}\nAccuracy: {accuracies[name]:.2%}", fontsize=TITLE_SIZE+4)
        ax.set_xlabel("PC 1")
        if i == 0:
            ax.set_ylabel("PC 2")
        ax.set_xlim(-max_range+1, max_range-1)
        ax.set_ylim(-max_range, max_range)
        ax.grid(True, alpha=0.4)
        ax.set_aspect('equal', 'box')

    # Add a single legend for the entire figure
    legend_elements = [
        Patch(facecolor=COLOR_PALETTE[0], label='Female'),
        Patch(facecolor=COLOR_PALETTE[1], label='Male')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 0.99), fontsize=LABEL_SIZE+3)
    
    plt.suptitle("Effect of Debiasing on Embedding Space and Gender Classification", fontsize=TITLE_SIZE + 6)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_plot("debiasing_comparison.png")

    # Print summary table
    print("\n" + "="*60)
    print("DEBIASING T-TEST SUMMARY")
    print("="*60)
    
    header = f"{'Attribute':<20} | {'Original t-stat':<20} | {'Hard Debias t-stat':<20} | {'Soft Debias t-stat':<20} | {'|Δ Hard|':<10} | {'|Δ Soft|':<10}"
    print(header)
    print("-"*len(header))

    for attribute in all_ttest_results["Original"].keys():
        original_t = all_ttest_results["Original"][attribute]['t_stat']
        hard_t = all_ttest_results["Hard Debias"][attribute]['t_stat']
        soft_t = all_ttest_results["Soft Debias(" + "\u03bb" + "=3.5)"][attribute]['t_stat']
        delta_hard = abs(original_t - hard_t)
        delta_soft = abs(original_t - soft_t)
        print(f"{attribute:<20} | {original_t:<20.2f} | {hard_t:<20.2f} | {soft_t:<20.2f} | {delta_hard:<10.2f} | {delta_soft:<10.2f}")
    
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
