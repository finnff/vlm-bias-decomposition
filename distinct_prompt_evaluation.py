import torch
import clip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
import random
from tqdm import tqdm
import os
import shutil
from scipy.stats import pointbiserialr
import gc


# NUMBER OF SAMPLES TO RUN CLIP ON
NUM_SAMPLES = 100000
DATALOADER_WORKERS = 16  # set to CPU cores -1
BATCH_SIZE = 256  # with 16 workers+prefetch=2 this requires ~40GB of RAM
PREFETCH_FACTOR = 2
USE_JIT = True  # JIT compilation helps on older GPUs

### FEATURE TOGGLES DISABLE TO SKIP CERTAIN ANALYSIS
TOGGLE_EXPLORE_DATASET = False
TOGGLE_VISUALIZE_SAMPLES = False
TOGGLE_CLIP_ANALYSIS = True
TOGGLE_EMBEDDING_ANALYSIS = True


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


if device == "cuda":
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

# using ViT-B here as CLIP examples on github also use that
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device, jit=USE_JIT)
model.eval()


class PreprocessedCelebA(torch.utils.data.Dataset):
    def __init__(self, base_dataset, preprocess):
        self.base = base_dataset
        self.preprocess = preprocess
        # Pre-compile the transforms if possible
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, attrs = self.base[index]
        # Keep as tensor if possible to avoid CPU-GPU transfers
        img_pil = self.to_pil(img)
        img_tensor = self.preprocess(img_pil)
        return img_tensor, attrs


class CelebAExplorer:
    def __init__(self, root_dir="./data", download=True, zip_threshold=1_000_000_000):
        """
        Initialize CelebA dataset explorer

        Args:
            root_dir: Directory to store/find the dataset
            download: Whether to download the dataset if not found
            zip_threshold: Minimum acceptable size (in bytes) for a valid zip
        """
        self.root_dir = root_dir
        self.dataset = None

        print("Loading CelebA dataset...")

        celeba_dir = os.path.join(root_dir, "celeba")
        zip_path = os.path.join(root_dir, "img_align_celeba.zip")

        # Clean up if the existing zip is too small (likely corrupted)
        if download and os.path.exists(zip_path):
            size = os.path.getsize(zip_path)
            if size < zip_threshold:
                print(
                    f"⚠️  Detected zip {zip_path} of size {size} bytes (< {zip_threshold} bytes). Removing corrupted zip..."
                )
                os.remove(zip_path)

        try:
            self.dataset = CelebA(
                root=root_dir,
                split="train",
                target_type="attr",
                download=download,
                transform=transforms.ToTensor(),
            )
            # besides attr we also have identity labels (name of celeb but
            # encoded into some number), and landmarks (keypoints on face)

            print(
                f"✓ Successfully loaded CelebA dataset with {len(self.dataset)} images"
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Troubleshooting steps:")
            print("1. Make sure you have enough disk space (CelebA is ~1.3GB)")
            print("2. Check your internet connection")
            print("3. Try deleting the ./data/celeba directory and re-running")
            print(
                "4. The download servers might be temporarily down, download manually from:"
            )
            print(
                "   https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ"
            )
            print("and place the zip in ./data/celeba")
            self.dataset = None

        self.attr_names = [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ]

    def explore_dataset(self, samples=1000):
        """Explore the CelebA dataset and show statistics"""
        print("\n" + "=" * 50)
        print("CELEBA DATASET EXPLORATION")
        print("=" * 50)

        # Basic statistics
        print(f"Total number of images: {len(self.dataset)}")
        print(f"Number of attributes: {len(self.attr_names)}")

        # Get a sample of attributes to analyze
        print("\nAnalyzing attributes distribution...")
        sample_size = max(1000, samples)  # Use sample for faster analysis
        sample_indices = random.sample(range(len(self.dataset)), sample_size)

        # Collect attribute statistics
        attr_counts = np.zeros(len(self.attr_names))

        for idx in tqdm(sample_indices[:samples], desc="Sampling attributes"):
            _, attrs = self.dataset[idx]
            attr_counts += attrs.numpy()

        attr_df = pd.DataFrame(
            {
                "Attribute": self.attr_names,
                "Count": attr_counts,
                "Percentage": (attr_counts / len(sample_indices[:samples]) * 100),
            }
        ).sort_values("Percentage", ascending=False)

        if TOGGLE_EXPLORE_DATASET:
            print("\n Attributes and how frequent they are:")
            print(attr_df.head(40).to_string(index=False))

        return attr_df

    def visualize_sample_images(self, num_images=8):
        """Visualize sample images with their attributes"""
        print(f"\nVisualizing {num_images} sample images...")

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for i in range(num_images):
            idx = random.randint(0, len(self.dataset) - 1)
            image, attrs = self.dataset[idx]

            # Convert tensor to PIL Image for display
            image_pil = transforms.ToPILImage()(image)

            # Get positive attributes
            positive_attrs = [
                self.attr_names[j] for j in range(len(attrs)) if attrs[j] == 1
            ]

            axes[i].imshow(image_pil)
            axes[i].axis("off")

            # Show top 10 attributes
            title = "\n".join(positive_attrs[:15])
            if len(positive_attrs) > 15:
                title += f"\n+{len(positive_attrs) - 15} more"
            axes[i].set_title(title, fontsize=11)

        plt.tight_layout()
        plt.savefig("celeba_samples.png", dpi=150, bbox_inches="tight")
        # uncommet to show plot, it gets saved to file regardless
        # plt.show()

    def run_improved_clip_analysis(
        self, num_samples=50, batch_size=128, num_workers=8, prefetch_factor=1
    ):
        """Run improved CLIP analysis with batch processing for speedup"""
        print(
            f"\nRunning improved CLIP analysis on {num_samples} images (batch size: {batch_size}, num_workers: {num_workers}, prefetch_factor: {prefetch_factor})..."
        )

        # instead of previous version, we now have binary prompts for certain attributes,
        # a person either is male or is female, has glasses or doesn't, etc.
        # this allows us to evaluate each of these attributes independently
        # This also tests every
        #
        # WARNING: because we check for each attribute separately,
        # this will take a long time to run, we could speed this up
        # by removing attributes that we are not interested in.
        prompt_groups = {
            "gender": ["a photo of a man", "a photo of a woman"],
            "binary_attributes": {
                "5_o_clock_shadow": (
                    "a person with a 5 o'clock shadow",
                    "a person with a clean shaven face",
                ),
                "arched_eyebrows": (
                    "a person with arched eyebrows",
                    "a person with straight eyebrows",
                ),
                "attractive": ("an attractive person", "a person"),
                "bags_under_eyes": (
                    "a person with bags under their eyes",
                    "a person without bags under their eyes",
                ),
                "bald": ("a bald person", "a person with hair"),
                "bangs": ("a person with bangs", "a person without bangs"),
                "big_lips": ("a person with big lips", "a person with thin lips"),
                "big_nose": ("a person with a big nose", "a person with a small nose"),
                "black_hair": (
                    "a person with black hair",
                    "a person without black hair",
                ),
                "blond_hair": ("a person with blonde hair", "a person with dark hair"),
                "blurry": ("a blurry photo of a person", "a clear photo of a person"),
                "brown_hair": (
                    "a person with brown hair",
                    "a person without brown hair",
                ),
                "bushy_eyebrows": (
                    "a person with bushy eyebrows",
                    "a person with thin eyebrows",
                ),
                "chubby": ("a chubby person", "a thin person"),
                "double_chin": (
                    "a person with a double chin",
                    "a person without a double chin",
                ),
                "eyeglasses": ("a person wearing glasses", "a person without glasses"),
                "goatee": ("a person with a goatee", "a person without a goatee"),
                "gray_hair": ("a person with gray hair", "a person without gray hair"),
                "heavy_makeup": (
                    "a person wearing heavy makeup",
                    "a person wearing no makeup",
                ),
                "high_cheekbones": (
                    "a person with high cheekbones",
                    "a person with low cheekbones",
                ),
                "male": ("a male person", "a female person"),
                "mouth_slightly_open": (
                    "a person with mouth slightly open",
                    "a person with mouth closed",
                ),
                "mustache": ("a person with a mustache", "a person without a mustache"),
                "narrow_eyes": ("a person with narrow eyes", "a person with wide eyes"),
                "no_beard": ("a person without a beard", "a person with a beard"),
                "oval_face": (
                    "a person with an oval face",
                    "a person with a round face",
                ),
                "pale_skin": ("a person with pale skin", "a person with tan skin"),
                "pointy_nose": (
                    "a person with a pointy nose",
                    "a person with a round nose",
                ),
                "receding_hairline": (
                    "a person with a receding hairline",
                    "a person with a full hairline",
                ),
                "rosy_cheeks": (
                    "a person with rosy cheeks",
                    "a person with normal cheeks",
                ),
                "sideburns": ("a person with sideburns", "a person without sideburns"),
                "smiling": ("a smiling person", "a person with a neutral expression"),
                "straight_hair": (
                    "a person with straight hair",
                    "a person with curly hair",
                ),
                "wavy_hair": ("a person with wavy hair", "a person with straight hair"),
                "wearing_earrings": (
                    "a person wearing earrings",
                    "a person not wearing earrings",
                ),
                "wearing_hat": ("a person wearing a hat", "a person not wearing a hat"),
                "wearing_lipstick": (
                    "a person wearing lipstick",
                    "a person not wearing lipstick",
                ),
                "wearing_necklace": (
                    "a person wearing a necklace",
                    "a person not wearing a necklace",
                ),
                "wearing_necktie": (
                    "a person wearing a necktie",
                    "a person not wearing a necktie",
                ),
                "young": ("a young person", "an older person"),
            },
            "attributes": [
                "a person with a 5 o'clock shadow",
                "a person with a clean shaven face",
                "a person with arched eyebrows",
                "a person with straight eyebrows",
                "an attractive person",
                "a person",
                "a person with bags under their eyes",
                "a person without bags under their eyes",
                "a bald person",
                "a person with hair",
                "a person with bangs",
                "a person without bangs",
                "a person with big lips",
                "a person with thin lips",
                "a person with a big nose",
                "a person with a small nose",
                "a person with black hair",
                "a person without black hair",
                "a person with blonde hair",
                "a person with dark hair",
                "a blurry photo of a person",
                "a clear photo of a person",
                "a person with brown hair",
                "a person without brown hair",
                "a person with bushy eyebrows",
                "a person with thin eyebrows",
                "a chubby person",
                "a thin person",
                "a person with a double chin",
                "a person without a double chin",
                "a person wearing glasses",
                "a person without glasses",
                "a person with a goatee",
                "a person without a goatee",
                "a person with gray hair",
                "a person without gray hair",
                "a person wearing heavy makeup",
                "a person wearing no makeup",
                "a person with high cheekbones",
                "a person with low cheekbones",
                "a male person",
                "a female person",
                "a person with mouth slightly open",
                "a person with mouth closed",
                "a person with a mustache",
                "a person without a mustache",
                "a person with narrow eyes",
                "a person with wide eyes",
                "a person without a beard",
                "a person with a beard",
                "a person with an oval face",
                "a person with a round face",
                "a person with pale skin",
                "a person with tan skin",
                "a person with a pointy nose",
                "a person with a round nose",
                "a person with a receding hairline",
                "a person with a full hairline",
                "a person with rosy cheeks",
                "a person with normal cheeks",
                "a person with sideburns",
                "a person without sideburns",
                "a smiling person",
                "a person with a neutral expression",
                "a person with straight hair",
                "a person with curly hair",
                "a person with wavy hair",
                "a person wearing earrings",
                "a person not wearing earrings",
                "a person wearing a hat",
                "a person not wearing a hat",
                "a person wearing lipstick",
                "a person not wearing lipstick",
                "a person wearing a necklace",
                "a person not wearing a necklace",
                "a person wearing a necktie",
                "a person not wearing a necktie",
                "a young person",
                "an older person",
            ],
        }

        # Prepare all prompts for tokenization
        all_prompts = prompt_groups["gender"].copy()
        for pos, neg in prompt_groups["binary_attributes"].values():
            all_prompts.extend([pos, neg])

        all_tokens = clip.tokenize(all_prompts).to(device)
        with torch.no_grad():
            all_text_features = model.encode_text(all_tokens)
            all_text_features = all_text_features / all_text_features.norm(
                dim=-1, keepdim=True
            )

        sample_indices = random.sample(range(len(self.dataset)), num_samples)
        subset = torch.utils.data.Subset(self.dataset, sample_indices)
        preprocessed_subset = PreprocessedCelebA(subset, preprocess)

        dataloader = torch.utils.data.DataLoader(
            preprocessed_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )

        results = []

        for batch_images, batch_attrs in tqdm(dataloader, desc="Processing batches"):
            # Batch image preprocessing (still needs to run on CPU due to PIL and transforms)
            # images = [preprocess(transforms.ToPILImage()(img)) for img in batch_images]
            # batch_tensor = torch.stack(images).to(device)

            # Faster way with Tensors and pinned memory
            batch_tensor = batch_images.to(device)

            with torch.no_grad():
                batch_features = model.encode_image(batch_tensor)
                batch_features = batch_features / batch_features.norm(
                    dim=-1, keepdim=True
                )
                similarities = 100.0 * batch_features @ all_text_features.T

            for j in range(batch_tensor.size(0)):
                idx = sample_indices.pop(0)
                attrs = batch_attrs[j]
                result = {
                    "image_idx": idx,
                    "actual_attributes": [
                        self.attr_names[k] for k in range(len(attrs)) if attrs[k] == 1
                    ],
                    "clip_scores": {},
                }

                img_sims = similarities[j]

                # Gender scores
                gender_sims = img_sims[:2].softmax(dim=-1)
                result["clip_scores"]["gender"] = {
                    "man": float(gender_sims[0]),
                    "woman": float(gender_sims[1]),
                }

                # Binary attribute scores
                prompt_idx = 2
                for attr_name in prompt_groups["binary_attributes"]:
                    pair_sims = img_sims[prompt_idx : prompt_idx + 2].softmax(dim=-1)
                    result["clip_scores"][attr_name] = float(pair_sims[0])
                    prompt_idx += 2

                results.append(result)

        return self.analyze_improved_results(results)

    def analyze_improved_results(self, results):
        """Analyze results with better metrics"""
        print("\nImproved CLIP Analysis Results:")

        # 1. Gender classification accuracy
        male_correct = sum(
            1
            for r in results
            if (
                "Male" in r["actual_attributes"]
                and r["clip_scores"]["gender"]["man"] > 0.5
            )
        )
        female_correct = sum(
            1
            for r in results
            if (
                "Male" not in r["actual_attributes"]
                and r["clip_scores"]["gender"]["woman"] > 0.5
            )
        )
        total_male = sum(1 for r in results if "Male" in r["actual_attributes"])
        total_female = len(results) - total_male

        print(f"\nGender Classification Accuracy:")
        print(
            f"Male accuracy: {male_correct}/{total_male} = {male_correct/total_male:.1%}"
        )
        print(
            f"Female accuracy: {female_correct}/{total_female} = {female_correct/total_female:.1%}"
        )

        # 2. Attribute detection performance using ROC-AUC style metrics
        attribute_mapping = {
            "eyeglasses": "Eyeglasses",
            "receding_hairline": "Receding_Hairline",
            "smiling": "Smiling",
            "blond_hair": "Blond_Hair",
            "heavy_makeup": "Heavy_Makeup",
            "5_o_clock_shadow": "5_o_Clock_Shadow",
            "arched_eyebrows": "Arched_Eyebrows",
            "attractive": "Attractive",
            "bags_under_eyes": "Bags_Under_Eyes",
            "bald": "Bald",
            "bangs": "Bangs",
            "big_lips": "Big_Lips",
            "big_nose": "Big_Nose",
            "black_hair": "Black_Hair",
            "blurry": "Blurry",
            "brown_hair": "Brown_Hair",
            "bushy_eyebrows": "Bushy_Eyebrows",
            "chubby": "Chubby",
            "double_chin": "Double_Chin",
            "goatee": "Goatee",
            "gray_hair": "Gray_Hair",
            "high_cheekbones": "High_Cheekbones",
            "male": "Male",
            "mouth_slightly_open": "Mouth_Slightly_Open",
            "mustache": "Mustache",
            "narrow_eyes": "Narrow_Eyes",
            "no_beard": "No_Beard",
            "oval_face": "Oval_Face",
            "pale_skin": "Pale_Skin",
            "pointy_nose": "Pointy_Nose",
            "rosy_cheeks": "Rosy_Cheeks",
            "sideburns": "Sideburns",
            "straight_hair": "Straight_Hair",
            "wavy_hair": "Wavy_Hair",
            "wearing_earrings": "Wearing_Earrings",
            "wearing_hat": "Wearing_Hat",
            "wearing_lipstick": "Wearing_Lipstick",
            "wearing_necklace": "Wearing_Necklace",
            "wearing_necktie": "Wearing_Necktie",
            "young": "Young",
        }

        print("\nAttribute Detection Performance:")

        print("\nMetrics:")
        print(
            "- Avg Score (Has): How confident CLIP is when the attribute is present (higher is better)"
        )
        print(
            "- Avg Score (Has not): How confident CLIP is when attribute is absent (lower is better)"
        )
        print(
            "- Discrimination: Difference between the two (positive = CLIP can detect this attribute)\n"
        )

        print(
            f"{'Attribute':<20} | {'Avg Score (Has)':>15} | {'Avg Score (Has not)':>19} | {'Discrimination':>14}"
        )
        print("-" * 76)

        all_results = []

        for clip_attr, celeba_attr in attribute_mapping.items():
            has_attr = [
                r["clip_scores"][clip_attr]
                for r in results
                if celeba_attr in r["actual_attributes"]
            ]
            no_attr = [
                r["clip_scores"][clip_attr]
                for r in results
                if celeba_attr not in r["actual_attributes"]
            ]

            if has_attr and no_attr:
                avg_has = np.mean(has_attr)
                avg_no = np.mean(no_attr)
                discrimination = avg_has - avg_no
                all_results.append((clip_attr, avg_has, avg_no, discrimination))

        # Sort by discrimination (descending - highest to lowest)
        all_results.sort(key=lambda x: x[3], reverse=True)

        # Print sorted results
        for clip_attr, avg_has, avg_no, discrimination in all_results:
            print(
                f"{clip_attr:<20} | {avg_has:>15.3f} | {avg_no:>19.3f} | {discrimination:>+14.3f}"
            )
        # 3. Confusion analysis for interesting cases
        print("\nInteresting Cases:")

        # Find images where CLIP strongly disagrees with labels
        for r in results[:5]:  # Just show first 5
            gender_pred = "man" if r["clip_scores"]["gender"]["man"] > 0.5 else "woman"
            actual_gender = "man" if "Male" in r["actual_attributes"] else "woman"

            if gender_pred != actual_gender:
                print(
                    f"\nImage {r['image_idx']}: CLIP predicts {gender_pred}, "
                    f"labeled as {actual_gender} (confidence: {max(r['clip_scores']['gender'].values()):.2f})"
                )

        return results

    def compute_embedding_similarities(
        self, num_samples=50, batch_size=128, num_workers=8, prefetch_factor=1
    ):
        """Compare CLIP and CelebA in embedding space"""
        print(f"\nComputing embedding similarities for {num_samples} images...")

        # Create text prompts for each CelebA attribute
        attribute_prompts = []
        for attr in self.attr_names:
            # Convert attribute name to natural language
            prompt = f"a person who is {attr.lower().replace('_', ' ')}"
            attribute_prompts.append(prompt)

        # Get CLIP embeddings for all attribute prompts
        text_tokens = clip.tokenize(attribute_prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Sample images and compute correlations
        sample_indices = random.sample(range(len(self.dataset)), num_samples)
        # Process in chunks
        chunk_size = 50000
        all_image_embeddings = []
        all_attribute_labels = []

        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_indices = sample_indices[chunk_start:chunk_end]

            # Subset and dataset
            subset = Subset(self.dataset, chunk_indices)
            dataset = PreprocessedCelebA(subset, preprocess)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
            )

            chunk_embeddings = []
            chunk_labels = []

            for batch_images, batch_attrs in tqdm(
                dataloader,
                desc=f"Chunk {chunk_start}-{chunk_end}",
                leave=False,
            ):
                batch_images = batch_images.to(device)
                with torch.no_grad():
                    batch_features = model.encode_image(batch_images)
                    batch_features = batch_features / batch_features.norm(
                        dim=-1, keepdim=True
                    )
                chunk_embeddings.append(batch_features.cpu().numpy())
                chunk_labels.append(batch_attrs.numpy())

            all_image_embeddings.append(np.vstack(chunk_embeddings))
            all_attribute_labels.append(np.vstack(chunk_labels))
            torch.cuda.empty_cache()

        # Combine all
        image_embeddings = np.vstack(all_image_embeddings)
        attribute_labels = np.vstack(all_attribute_labels)
        similarities = image_embeddings @ text_features.cpu().numpy().T

        # Analyze correlation for each attribute
        print("\nCLIP-CelebA Attribute Correlations:")

        print("\nMetrics:")
        print(
            "- Correlation: How well CLIP's similarity scores align with CelebA labels (-1 to +1, higher (absolute) magnitude = better)"
        )
        print(
            "- Mean Sim (WITH): Average cosine similarity (dot product of normalized embeddings) between images WITH the attribute and its text description"
        )
        print(
            "- Mean Sim (WITHOUT): Average cosine similarity between images WITHOUT the attribute and its text description"
        )
        print(
            "- Higher correlation means CLIP understands this attribute; larger gap between WITH/WITHOUT is better\n"
        )

        print(
            f"{'Attribute':<20} | {'Correlation':>12} | {'Mean Sim (WITH)':>15} | {'Mean Sim (WITHOUT)':>19}"
        )
        print("-" * 76)

        correlations = []
        for i, attr_name in enumerate(self.attr_names):
            has_attr = attribute_labels[:, i] == 1

            if has_attr.sum() > 0 and (~has_attr).sum() > 0:
                corr, _ = pointbiserialr(has_attr, similarities[:, i])
                mean_has = similarities[has_attr, i].mean()
                mean_hasnt = similarities[~has_attr, i].mean()

                correlations.append((attr_name, corr, mean_has, mean_hasnt))

        # Sort by correlation strength
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for attr_name, corr, mean_has, mean_hasnt in correlations[:40]:  # Top 40
            print(
                f"{attr_name:<20} | {corr:>12.3f} | {mean_has:>15.3f} | {mean_hasnt:>19.3f}"
            )


def main():
    # Set random seeds for reproducibility
    random.seed(48)
    torch.manual_seed(48)
    np.random.seed(48)

    # Initialize the analyzer
    analyzer = CelebAExplorer(root_dir="./data", download=True)

    # Print dataset info
    print(f"\nDataset loaded with {len(analyzer.dataset)} images")
    print(f"Number of attributes: {len(analyzer.attr_names)}")
    print(f"Attributes: {', '.join(analyzer.attr_names[:10])}...")

    # Visualize sample images
    if TOGGLE_VISUALIZE_SAMPLES:
        analyzer.visualize_sample_images(num_images=8)

    # Run the improved CLIP analysis with binary comparisons
    if TOGGLE_CLIP_ANALYSIS:
        print("\n" + "=" * 76)
        print("IMPROVED CLIP ANALYSIS WITH BINARY COMPARISONS")
        print("=" * 76)
        improved_results = analyzer.run_improved_clip_analysis(
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            num_workers=DATALOADER_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )

    # Run the embedding similarity analysis
    if TOGGLE_EMBEDDING_ANALYSIS:
        print("\n" + "=" * 76)
        print("EMBEDDING SPACE CORRELATION ANALYSIS")
        print("=" * 76)
        analyzer.compute_embedding_similarities(
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            num_workers=DATALOADER_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )

    # Save results for later analysis
    print("\nSaving analysis results...")
    import json

    # Save improved results
    with open("clip_celeba_improved_results.json", "w") as f:
        # Convert results to serializable format
        serializable_results = []
        for r in improved_results:
            serializable_results.append(
                {
                    "image_idx": r["image_idx"],
                    "actual_attributes": r["actual_attributes"],
                    "clip_scores": r["clip_scores"],
                }
            )
        json.dump(serializable_results, f, indent=2)

    print("Analysis complete! Results saved to 'clip_celeba_improved_results.json'")


if __name__ == "__main__":
    main()
