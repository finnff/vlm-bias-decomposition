import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def load_celeba_dataset(root="./data", split="train", download=False):
    dataset = CelebA(
        root=root,
        split=split,
        target_type="attr",
        transform=preprocess,
        download=download
    )
    return dataset


def get_labels(dataset, batch_size=64, max_samples=5000):
    male_idx = dataset.attr_names.index("Male")
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


def get_all_embeddings_and_attrs(dataset, batch_size=64, max_samples=5000):
    attr_names = dataset.attr_names  
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

    return clip_embs, attr_mat, idx_list, attr_names
