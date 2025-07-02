import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset
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


def get_labels(dataset, batch_size=64, max_samples=100000):
    male_idx = dataset.attr_names.index("Male")
    
    # Limit the dataset first to avoid unneeded processing
    limited_dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    
    dataloader = DataLoader(limited_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    clip_embeddings = []
    gender_labels = []
    image_list = []

    for images, attrs in tqdm(dataloader, total=len(limited_dataset)//batch_size):
        labels = attrs[:, male_idx].int()
        images = images.to(device)

        with torch.no_grad():
            features = model.encode_image(images).cpu()

        clip_embeddings.append(features)
        gender_labels.append(labels)
        
        # Optional: Comment this out if you don't really need the images
        # image_list.extend(images.cpu())

    clip_embeddings = torch.cat(clip_embeddings, dim=0)
    gender_labels = torch.cat(gender_labels, dim=0)

    return clip_embeddings, gender_labels, image_list


def get_all_embeddings_and_attrs(dataset, batch_size=64, max_samples=100000):
    attr_names = dataset.attr_names
    limited_dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    
    loader = DataLoader(
        limited_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    clip_embs = []
    attr_list = []
    idx_list = []

    for batch_idx, (images, attrs) in enumerate(tqdm(loader, total=len(limited_dataset) // batch_size)):
        images = images.to(device)
        labels = attrs.int()  # [batch, 40]

        with torch.no_grad():
            feats = model.encode_image(images).cpu()  # [batch, 512]

        clip_embs.append(feats)
        attr_list.append(labels)

        # dataset indices from Subset are consecutive from 0
        start_idx = batch_idx * batch_size
        current_batch_size = images.size(0)
        idx_list.extend(range(start_idx, start_idx + current_batch_size))

    clip_embs = torch.cat(clip_embs, dim=0)           # [N, 512]
    attr_mat = torch.cat(attr_list, dim=0).numpy()    # [N, 40]
    idx_list = idx_list[:clip_embs.shape[0]]          # Trim just in case

    return clip_embs, attr_mat, idx_list, attr_names
