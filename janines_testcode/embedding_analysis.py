import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_eval(clip_embeddings, gender_labels):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(clip_embeddings.numpy())

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=gender_labels.numpy(),
                    palette=["red", "blue"], alpha=0.6, edgecolor=None)
    plt.title("PCA of CLIP Embeddings Colored by Gender")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(labels=["Female", "Male"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gender_pca.png")


def tsne_eval(clip_embeddings, gender_labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(clip_embeddings.numpy())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=gender_labels.numpy(),
                    palette=["red", "blue"], alpha=0.6)
    plt.title("t-SNE of CLIP Embeddings Colored by Gender")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(labels=["Female", "Male"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gender_tsne.png")


def plot_attribute_bias_directions(clip_embeddings, attr_matrix, attr_names):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(clip_embeddings.numpy())  # shape [N,2]

    #For each attribute, compute negative‐class & positive‐class centroids

    #store μ⁻, μ⁺ in two lists
    num_attrs = len(attr_names)
    neg_centroids = np.zeros((num_attrs, 2), dtype=float)
    pos_centroids = np.zeros((num_attrs, 2), dtype=float)

    for a_idx in range(num_attrs):
        mask_neg = (attr_matrix[:, a_idx] == 0)
        mask_pos = (attr_matrix[:, a_idx] == 1)

        if mask_neg.sum() < 10 or mask_pos.sum() < 10:
            neg_centroids[a_idx] = np.nan
            pos_centroids[a_idx] = np.nan
            continue

        neg_centroids[a_idx] = X_2d[mask_neg].mean(axis=0)
        pos_centroids[a_idx] = X_2d[mask_pos].mean(axis=0)

    #Plot all “bias direction” lines in 2D PCA space
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    #For each attribute, draw a line from μ⁻ to μ⁺
    for a_idx, name in enumerate(attr_names):
        neg_c = neg_centroids[a_idx]
        pos_c = pos_centroids[a_idx]
        if np.isnan(neg_c).any() or np.isnan(pos_c).any():
            continue

        ax.plot(
            [neg_c[0], pos_c[0]],
            [neg_c[1], pos_c[1]],
            linewidth=1.5,
            alpha=0.6,
            label=name
        )

        midpt = (neg_c + pos_c) / 2
        ax.text(
            midpt[0],
            midpt[1],
            name,
            fontsize=8,
            alpha=0.7,
            horizontalalignment="center",
            verticalalignment="center"
        )

    ax.set_title("CelebA Attribute Bias Directions in 2D PCA of CLIP Embeddings")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)
    ax.set_aspect("equal", "box")

    # ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig("attribute_bias_directions.png", dpi=200)
    #plt.show()
