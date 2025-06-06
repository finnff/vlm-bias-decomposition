import torch
import matplotlib.pyplot as plt


def show_misclassified(images, y_true, y_pred, indices, n=5):
    plt.figure(figsize=(15, 3))

    for i, idx in enumerate(indices[:n]):
        plt.subplot(1, n, i + 1)

        img = images[idx]
        # clip ViT-B/32 uses mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
        img = img * std + mean
        img = img.clamp(0,1)

        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"True: {'Male' if y_true[idx] else 'Female'}\nPred: {'Male' if y_pred[idx] else 'Female'}")
        plt.axis("off")

    plt.suptitle(f"Misclassified Gender Images (Showing {n})")
    plt.tight_layout()
    plt.savefig("misclasvis.png")