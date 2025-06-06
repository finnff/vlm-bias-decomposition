import torch
import torchvision.transforms as T
from torchvision.datasets import CelebA
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = CelebA(
    root="./data",
    split="train",
    target_type="attr",
    transform=preprocess,
    download=False
)


BATCH_SIZE = 64
MAX_SAMPLES = 5000   #small subset

attr_names = dataset.attr_names  
num_attrs = len(attr_names)      #should be 40

"""
    Returns:
    clip_emb:   torch.FloatTensor of shape [N,512]
    attr_mat:   numpy array of shape [N,40] (0/1)
    indices:    list of length N, storing the integer index within the dataset
"""
def get_all_embeddings_and_attrs(dataset, batch_size=64, max_samples=5000):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    clip_embs = []
    attr_list = []
    idx_list = []

    total_seen = 0
    for batch_idx, (images, attrs) in enumerate(tqdm(loader)):
        if total_seen >= max_samples:
            break

        current_batch_size = images.size(0)
        labels = attrs[:, :].int()  # shape [batch, 40]

        images = images.to(device)
        with torch.no_grad():
            feats = model.encode_image(images).cpu()  # [batch,512]
        clip_embs.append(feats)
        attr_list.append(labels)

        # store which dataset indices these came from
        start_idx = batch_idx * batch_size
        for offset in range(current_batch_size):
            idx_list.append(start_idx + offset)

        total_seen += current_batch_size

    clip_embs = torch.cat(clip_embs, dim=0)           # [N,512]
    attr_mat = torch.cat(attr_list, dim=0).numpy()    # [N,40]
    idx_list = idx_list[: clip_embs.shape[0]]         # length N

    return clip_embs, attr_mat, idx_list


# Extract (N,512) and (N,40)
clip_embeddings, attr_matrix, indices = get_all_embeddings_and_attrs(
    dataset, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES
)
N = clip_embeddings.shape[0]
print(f"Extracted embeddings for N={N} images; attr_matrix shape = {attr_matrix.shape}")
# attr_matrix[i,j] ∈ {0,1} indicates attribute j for image i

#Run PCA once on all embeddings
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(clip_embeddings.numpy())  # shape [N,2]




#For each attribute, compute negative‐class & positive‐class centroids

#store μ⁻, μ⁺ in two lists
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
plt.show()

distances = np.linalg.norm(pos_centroids - neg_centroids, axis=1)  # [40]

attr_dist_pairs = list(zip(attr_names, distances))
attr_dist_pairs.sort(key=lambda pair: pair[1], reverse=True)

print("Top 10 attributes by CLIP‐embedding PCA separation:")
for i in range(10):
    name, dist = attr_dist_pairs[i]
    print(f"  {i+1:2d}. {name:<20s}  → distance = {dist:.4f}")

male_idx = 20
mu_female = neg_centroids[male_idx]   # [2] array for "Male = 0"
mu_male   = pos_centroids[male_idx]   # [2] array for "Male = 1"
v_mf = mu_male - mu_female            # 2D vector pointing Female→Male
norm_vmf = np.linalg.norm(v_mf)

#or each attribute, compute v_A = μ^+_A – μ^-_A
dot_projs = []
for a_idx, name in enumerate(attr_names):
    vA = pos_centroids[a_idx] - neg_centroids[a_idx]
    proj = np.dot(vA, v_mf) / (norm_vmf + 1e-8)#projection onto gender axis
    dot_projs.append((name, proj))

#most male‐aligned first
dot_projs.sort(key=lambda pair: pair[1], reverse=True)


print("Top 5 Male‐aligned attributes (highest positive projection onto male axis):")
for i in range(5):
    name, value = dot_projs[i]
    print(f"  {i+1:2d}. {name:<20s}  proj = {value:.4f}")



print("\nTop 5 Female‐aligned attributes (most negative projection onto male axis):")
for i in range(5):
    name, value = dot_projs[-(i+1)]
    print(f"  {i+1:2d}. {name:<20s}  proj = {value:.4f}")




"""
SO in the last thing we found a top five of female attributes, now i claim, no beard, attractive and young are not female related.
plan:
1 find males with all those attributes and males without all those attributes
2 predict male/female on those 2 male versions
3 calculate average confidence for the 2
4 t-test
"""

#need a classifier for that this is super double
male_idx = attr_names.index("Male")
gender_labels = attr_matrix[:, male_idx] 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X_train, X_test, y_train, y_test = train_test_split(
    clip_embeddings.numpy(),    
    gender_labels,               
    test_size=0.2,
    random_state=42,
    stratify=gender_labels
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print(f"Gender classifier training accuracy: {clf.score(X_train, y_train):.4f}")
print(f"Gender classifier test accuracy:     {clf.score(X_test, y_test):.4f}")



male_idx        = attr_names.index("Male")
no_beard_idx    = attr_names.index("No_Beard")
attractive_idx  = attr_names.index("Attractive")
young_idx       = attr_names.index("Young")

#supposedly female attributes
mask_group1 = (
    (attr_matrix[:, male_idx] == 1) &
    (attr_matrix[:, no_beard_idx] == 1) &
    (attr_matrix[:, attractive_idx] == 1) &
    (attr_matrix[:, young_idx] == 1)
)
idxs_group1 = np.where(mask_group1)[0]

mask_group2 = (
    (attr_matrix[:, male_idx] == 1) &
    (attr_matrix[:, no_beard_idx] == 0) &
    (attr_matrix[:, attractive_idx] == 0) &
    (attr_matrix[:, young_idx] == 0)
)
idxs_group2 = np.where(mask_group2)[0]

print(f"\nGroup 1 (male & no_beard & attractive & young):  {len(idxs_group1)} examples")
print(f"Group 2 (male & no no_beard & no attractive & no young):  {len(idxs_group2)} examples\n")

#CLIP embeddings for each group
X_group1 = clip_embeddings[idxs_group1].numpy()
X_group2 = clip_embeddings[idxs_group2].numpy() 


probs_group1 = clf.predict_proba(X_group1)[:, 1]
probs_group2 = clf.predict_proba(X_group2)[:, 1]  



avg1 = probs_group1.mean()
std1 = probs_group1.std()
avg2 = probs_group2.mean()
std2 = probs_group2.std()
print(f"Average P(male) for Group 1 (female‐associated attributes)  = {avg1:.4f}  ± {std1:.4f}")
print(f"Average P(male) for Group 2 (none of those attributes)       = {avg2:.4f}  ± {std2:.4f}\n")


#t‐test:
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(probs_group1, probs_group2, equal_var=False)
print(f"Two‐sample t‐test: t = {t_stat:.2f},  p = {p_val:.3e}")

""""
RESULT: 

Average P(male) for Group 1 (female‐associated attributes)  = 0.9847  ± 0.0545
Average P(male) for Group 2 (none of those attributes)       = 0.9983  ± 0.0091

Two‐sample t‐test: t = -4.29,  p = 2.359e-05

yas found a statistic significant difference :D
"""