import torch
import torchvision.transforms as T
from torchvision.datasets import CelebA
from PIL import Image
import clip
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = CelebA(
    root="./data", split="train", target_type="attr",transform=preprocess, download=False
)

#20
male_idx = dataset.attr_names.index("Male")


def get_labels(dataset, batch_size=64, max_samples=5000):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    clip_embeddings = []
    gender_labels = []
    image_list = []  # store raw PIL images for visualization

    for i, (images, attrs) in enumerate(tqdm(dataloader)):
        if i * batch_size >= max_samples:
            break

        labels = attrs[:, male_idx].int()
        images = images.to(device)

        with torch.no_grad():
            features = model.encode_image(images).cpu()

        clip_embeddings.append(features)
        gender_labels.append(labels)
        image_list.extend(images.cpu())  # Store images (normalized tensors)

    clip_embeddings = torch.cat(clip_embeddings, dim=0)
    gender_labels = torch.cat(gender_labels, dim=0)

    return clip_embeddings, gender_labels, image_list


def train(clip_embeddings, gender_labels):
    X_train, X_test, y_train, y_test = train_test_split(
    clip_embeddings.numpy(), gender_labels.numpy(), test_size=0.2, random_state=42
)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    misclassified = (y_pred != y_test)

    # Get image indices from test set
    misclassified_indices = np.where(misclassified)[0]
    return X_test, y_test, y_pred, misclassified_indices


def eval(y_test, y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def pca_eval(clip_embeddings):
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

def tsne_eval(clip_embeddings):
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

def show_misclassified(images, y_true, y_pred, indices, n=5):
    plt.figure(figsize=(15, 3))

    for i, idx in enumerate(indices[:n]):
        plt.subplot(1, n, i + 1)

        img = images[idx]  # shape: [3, 224, 224] - torch.Tensor
        img = img.clone()  # avoid modifying in-place

        # Unnormalize CLIP input
        mean = torch.tensor([0.4814, 0.4578, 0.4082]).view(3, 1, 1)
        std = torch.tensor([0.2686, 0.2613, 0.2758]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        img = img.permute(1, 2, 0).numpy()  # CHW → HWC

        plt.imshow(img)
        plt.title(f"True: {'Male' if y_true[idx] else 'Female'}\nPred: {'Male' if y_pred[idx] else 'Female'}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_examples.png")


clip_embeddings, gender_labels, paths = get_labels(dataset, batch_size=64, max_samples=5000)
pca_eval(clip_embeddings)
tsne_eval(clip_embeddings)
X_test, y_test, y_pred, mis = train(clip_embeddings, gender_labels)
eval(y_test, y_pred)
show_misclassified(paths, y_test, y_pred, mis, n=5)

