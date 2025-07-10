

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Define font sizes for consistent plotting
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

def plot_t_statistic_deltas(df, output_dir):
    """Generates a 2x2 plot of t-statistic deltas, with each subplot sorted and clipped."""
    print("\n--- Generating Sorted & Clipped t-statistic Delta Plot ---")
    
    df['Δ Soft_abs'] = df["Δ Soft"].abs()
    df['Δ Hard_abs'] = df["Δ Hard"].abs()
    df['Δ Soft%_abs'] = df["Δ Soft%"].abs()
    df['Δ Hard%_abs'] = df["Δ Hard%"].abs()

    fig, axs = plt.subplots(2, 2, figsize=(22, 16))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plot_configs = {
        (0, 0): ('Δ Soft_abs', '|Δ Soft| Change in t-statistic', 'skyblue', None),
        (0, 1): ('Δ Hard_abs', '|Δ Hard| Change in t-statistic', 'salmon', None),
        (1, 0): ('Δ Soft%_abs', '|Δ Soft %| Change in t-statistic', 'skyblue', None),
        (1, 1): ('Δ Hard%_abs', '|Δ Hard %| Change in t-statistic', 'salmon', 6000),
    }

    for (row, col), (sort_key, title, color, clip_val) in plot_configs.items():
        ax = axs[row, col]
        df_sorted = df.sort_values(by=sort_key, ascending=True)
        y_pos = np.arange(len(df_sorted))
        
        ax.barh(y_pos, df_sorted[sort_key], align='center', color=color)
        
        if clip_val:
            ax.set_xlim(right=clip_val)
            ax.set_title(title + f' (View Clipped at {clip_val})', fontsize=TITLE_FONTSIZE)
        else:
            ax.set_title(title, fontsize=TITLE_FONTSIZE)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['Attribute'], fontsize=TICK_FONTSIZE)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)

    plt.suptitle('Absolute Debiasing Efficacy Analysis (t-statistic, Sorted)', fontsize=22, y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, "debiasing_analysis_deltas_final.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_cohen_d_reduction(df, output_dir):
    """Calculates and plots the Cohen's d reduction metric in separate, sorted subplots with view clipping."""
    print("\n--- Plotting Cohen's d Reduction Metric (Subplots, Clipped View) ---")
    
    df['soft_reduction'] = (abs(df["Baseline C'd"]) - abs(df["Soft-Debias C'd"])) / abs(df["Baseline C'd"])
    df['hard_reduction'] = (abs(df["Baseline C'd"]) - abs(df["Hard-Debias C'd"])) / abs(df["Baseline C'd"])
    
    reduction_df = df[['Attribute', 'soft_reduction', 'hard_reduction']]
    print("Cohen's d Reduction Metric ((|base|-|debiased|)/|base|):\n" + reduction_df.to_string(index=False))
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 12), sharey=False)

    # Soft-Debias Subplot
    df_sorted_soft = df.sort_values(by='soft_reduction', ascending=True)
    y_pos_soft = np.arange(len(df_sorted_soft))
    axs[0].barh(y_pos_soft, df_sorted_soft['soft_reduction'], align='center', color='skyblue')
    axs[0].axvline(x=1.0, color='green', linestyle='--', label='Perfect Elimination (1.0)')
    axs[0].axvline(x=0.0, color='black', linestyle='--', label='No Improvement (0.0)')
    axs[0].set_title('Soft-Debias Cohen\'s d Reduction', fontsize=TITLE_FONTSIZE)
    axs[0].set_yticks(y_pos_soft)
    axs[0].set_yticklabels(df_sorted_soft['Attribute'], fontsize=TICK_FONTSIZE)
    axs[0].legend()

    # Hard-Debias Subplot
    df_sorted_hard = df.sort_values(by='hard_reduction', ascending=True)
    y_pos_hard = np.arange(len(df_sorted_hard))
    axs[1].barh(y_pos_hard, df_sorted_hard['hard_reduction'], align='center', color='salmon')
    axs[1].axvline(x=1.0, color='green', linestyle='--')
    axs[1].axvline(x=0.0, color='black', linestyle='--')
    axs[1].set_title('Hard-Debias Cohen\'s d Reduction (View [-1, 1])', fontsize=TITLE_FONTSIZE)
    axs[1].set_yticks(y_pos_hard)
    axs[1].set_yticklabels(df_sorted_hard['Attribute'], fontsize=TICK_FONTSIZE)
    axs[1].set_xlim(-1, 1.1) # Set the x-axis view

    for ax in axs:
        ax.set_xlabel("Cohen's d Reduction", fontsize=LABEL_FONTSIZE)
        ax.grid(True, axis='x', linestyle='--')

    plt.suptitle("Cohen's d Reduction by Method (Sorted)", fontsize=22, y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.join(output_dir, "debiasing_cohen_d_reduction_final.png")
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")
    plt.close()

def plot_combined_efficacy_score(df, output_dir):
    """Calculates and plots a combined efficacy score in separate, sorted subplots."""
    print("\n--- Plotting Sorted Combined Efficacy Score (Subplots) ---")

    soft_br = (abs(df["Baseline C'd"]) - abs(df["Soft-Debias C'd"])) / abs(df["Baseline C'd"])
    hard_br = (abs(df["Baseline C'd"]) - abs(df["Hard-Debias C'd"])) / abs(df["Baseline C'd"])
    soft_br[soft_br < 0] = 0
    hard_br[hard_br < 0] = 0

    soft_ap = df["Soft-Debias Acc"] / df["Baseline Acc"]
    hard_ap = df["Hard-Debias Acc"] / df["Baseline Acc"]
    soft_ap[soft_ap > 1] = 1
    hard_ap[hard_ap > 1] = 1

    df['Soft Efficacy'] = soft_br * soft_ap
    df['Hard Efficacy'] = hard_br * hard_ap
    
    print("Combined Efficacy Score (Bias Reduction * Accuracy Preservation):\n" + df[['Attribute', 'Soft Efficacy', 'Hard Efficacy']].to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(20, 12), sharey=False)

    # Soft-Debias Subplot
    df_sorted_soft = df.sort_values(by='Soft Efficacy', ascending=True)
    y_pos_soft = np.arange(len(df_sorted_soft))
    axs[0].barh(y_pos_soft, df_sorted_soft['Soft Efficacy'], align='center', color='skyblue')
    axs[0].set_title('Soft-Debias Efficacy', fontsize=TITLE_FONTSIZE)
    axs[0].set_yticks(y_pos_soft)
    axs[0].set_yticklabels(df_sorted_soft['Attribute'], fontsize=TICK_FONTSIZE)

    # Hard-Debias Subplot
    df_sorted_hard = df.sort_values(by='Hard Efficacy', ascending=True)
    y_pos_hard = np.arange(len(df_sorted_hard))
    axs[1].barh(y_pos_hard, df_sorted_hard['Hard Efficacy'], align='center', color='salmon')
    axs[1].set_title('Hard-Debias Efficacy', fontsize=TITLE_FONTSIZE)
    axs[1].set_yticks(y_pos_hard)
    axs[1].set_yticklabels(df_sorted_hard['Attribute'], fontsize=TICK_FONTSIZE)

    for ax in axs:
        ax.set_xlabel('Efficacy Score', fontsize=LABEL_FONTSIZE)
        ax.grid(True, axis='x', linestyle='--')

    plt.suptitle('Combined Debiasing Efficacy Score (Sorted)', fontsize=22, y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.join(output_dir, "debiasing_efficacy_score_final.png")
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")
    plt.close()

def plot_cohen_d_change_vector(df, output_dir):
    """
    Generates a dumbbell plot showing the change in Cohen's d from baseline
    to debiased, clearly visualizing magnitude and direction changes.
    """
    print("\n--- Generating Cohen's d Change Vector Plot ---")
    
    # Sort by the magnitude of the initial bias to prioritize the most biased attributes
    df_sorted = df.sort_values(by="Baseline C'd", key=abs, ascending=True)
    
    attributes = df_sorted['Attribute']
    y_pos = np.arange(len(attributes))

    fig, axs = plt.subplots(1, 2, figsize=(20, 14), sharey=True)
    fig.suptitle("Cohen's d Before and After Debiasing (Sorted by Baseline Bias)", fontsize=22, y=0.97)

    # --- Soft Debias Subplot ---
    ax_soft = axs[0]
    baseline_d = df_sorted["Baseline C'd"]
    soft_debiased_d = df_sorted["Soft-Debias C'd"]
    
    ax_soft.hlines(y=y_pos, xmin=baseline_d, xmax=soft_debiased_d, color='lightgray', alpha=0.9, zorder=1)
    ax_soft.scatter(baseline_d, y_pos, color='gray', s=50, label='Baseline d', zorder=2, marker='o')
    ax_soft.scatter(soft_debiased_d, y_pos, color='skyblue', s=50, label='Soft-Debiased d', zorder=2, marker='X')
    ax_soft.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero Bias')
    ax_soft.set_title("Soft-Debias: Change in Cohen's d", fontsize=TITLE_FONTSIZE)
    ax_soft.legend()
    
    # --- Hard Debias Subplot ---
    ax_hard = axs[1]
    hard_debiased_d = df_sorted["Hard-Debias C'd"]
    
    ax_hard.hlines(y=y_pos, xmin=baseline_d, xmax=hard_debiased_d, color='lightgray', alpha=0.9, zorder=1)
    ax_hard.scatter(baseline_d, y_pos, color='gray', s=50, label='Baseline d', zorder=2, marker='o')
    ax_hard.scatter(hard_debiased_d, y_pos, color='salmon', s=50, label='Hard-Debiased d', zorder=2, marker='X')
    ax_hard.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero Bias')
    
    view_limit = max(abs(df_sorted["Baseline C'd"]).max(), 2.0)
    ax_hard.set_xlim(-view_limit, view_limit)
    ax_hard.set_title(f"Hard-Debias: Change in Cohen's d (View Clipped to ±{view_limit:.1f})", fontsize=TITLE_FONTSIZE)
    ax_hard.legend()

    # Common formatting
    for ax in axs:
        ax.set_xlabel("Cohen's d", fontsize=LABEL_FONTSIZE)
        ax.grid(True, axis='x', linestyle='--')

    plt.yticks(y_pos, attributes, fontsize=TICK_FONTSIZE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, "debiasing_cohen_d_change_vector.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def main(csv_path):
    """Main function to run all analyses."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'")
        sys.exit(1)

    output_dir = os.path.dirname(csv_path)
    if not output_dir:
        output_dir = '.'

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    df['Δ Soft'] = df["Soft-Debias t"] - df["Baseline t"]
    df['Δ Hard'] = df["Hard-Debias t"] - df["Baseline t"]
    df['Δ Soft%'] = (df['Δ Soft'] / df["Baseline t"].replace(0, 1)) * 100
    df['Δ Hard%'] = (df['Δ Hard'] / df["Baseline t"].replace(0, 1)) * 100
    
    # --- Generate All Plots ---
    plot_t_statistic_deltas(df.copy(), output_dir)
    plot_cohen_d_reduction(df.copy(), output_dir)
    plot_combined_efficacy_score(df.copy(), output_dir)
    plot_cohen_d_change_vector(df.copy(), output_dir)

    print("\nAll analyses complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_and_visualize_deltas.py <path_to_csv_file>")
        sys.exit(1)
    
    main(sys.argv[1])
