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
import gc


# NUMBER OF SAMPLES TO RUN CLIP ON
NUM_SAMPLES = 10000
DATALOADER_WORKERS = 8  # set to CPU cores -1 to prevent crash
BATCH_SIZE = 256
PREFETCH_FACTOR = 1
USE_JIT = True  # JIT compilation helps on older GPUs


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

    def run_clip_analysis(
        self, num_samples=50, batch_size=128, num_workers=4, prefetch_factor=1
    ):
        """Run CLIP analysis on a subset of CelebA images"""
        print(f"\nRunning CLIP analysis on {num_samples} images...")

        # Define text prompts for CLIP
        text_prompts = [
            "a photo of a man with a receding hairline",
            "a photo of a man with glasses",
            "a photo of an attractive man",
            "a photo of a woman smiling",
            "a photo of a woman with makeup",
            "a photo of a woman with blonde hair",
        ]

        # Tokenize text prompts
        text_tokens = clip.tokenize(text_prompts).to(device)

        # Sample random images
        sample_indices = random.sample(range(len(self.dataset)), num_samples)
        subset = Subset(self.dataset, sample_indices)

        dataloader = DataLoader(
            PreprocessedCelebA(subset, preprocess),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )

        results = []
        index_cursor = 0

        for batch_images, batch_attrs in tqdm(dataloader, desc="Processing batches"):
            batch_images = batch_images.to(device)

            with torch.no_grad():
                logits_per_image, _ = model(batch_images, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            for i in range(batch_images.size(0)):
                idx = sample_indices[index_cursor]
                attrs = batch_attrs[i]
                index_cursor += 1

                results.append(
                    {
                        "image_idx": idx,
                        "clip_predictions": dict(zip(text_prompts, probs[i])),
                        "actual_attributes": [
                            self.attr_names[j]
                            for j in range(len(attrs))
                            if attrs[j] == 1
                        ],
                        "top_clip_prediction": text_prompts[np.argmax(probs[i])],
                        "top_clip_confidence": float(np.max(probs[i])),
                    }
                )

        return self.analyze_clip_results(results, text_prompts)

    # code below was generated by AI its super verbose and not DRY, might be worth writing a function with dict here if
    # we are going to use this more??
    def analyze_clip_results(self, results, text_prompts):
        print("\nAnalyzing CLIP results...")

        avg_confidences = {}
        for prompt in text_prompts:
            confidences = [r["clip_predictions"][prompt] for r in results]
            avg_confidences[prompt] = np.mean(confidences)

        print("\nAverage CLIP confidences by text prompt:")
        for prompt, conf in sorted(
            avg_confidences.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{prompt}: {conf:.3f}")

        print("\nCorrelation analysis:")

        male_images = [r for r in results if "Male" in r["actual_attributes"]]
        female_images = [r for r in results if "Male" not in r["actual_attributes"]]

        if male_images and female_images:
            man_prompts = [p for p in text_prompts if "man" in p.lower()]
            woman_prompts = [p for p in text_prompts if "woman" in p.lower()]

            if man_prompts:
                male_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in man_prompts])
                        for r in male_images
                    ]
                )
                print(
                    f"Male images - average 'man' prompts confidence: {male_conf:.3f}"
                )

            if woman_prompts:
                female_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in woman_prompts])
                        for r in female_images
                    ]
                )
                print(
                    f"Female images - average 'woman' prompts confidence: {female_conf:.3f}"
                )

        young_images = [r for r in results if "Young" in r["actual_attributes"]]
        old_images = [r for r in results if "Young" not in r["actual_attributes"]]

        if young_images and old_images:
            young_prompts = [p for p in text_prompts if "young" in p.lower()]
            old_prompts = [p for p in text_prompts if "old" in p.lower()]

            if young_prompts:
                young_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in young_prompts])
                        for r in young_images
                    ]
                )
                print(f"Young images - 'young' prompts confidence: {young_conf:.3f}")

            if old_prompts:
                old_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in old_prompts])
                        for r in old_images
                    ]
                )
                print(f"Non-young images - 'old' prompts confidence: {old_conf:.3f}")

        glasses_images = [r for r in results if "Eyeglasses" in r["actual_attributes"]]
        no_glasses_images = [
            r for r in results if "Eyeglasses" not in r["actual_attributes"]
        ]

        if glasses_images and no_glasses_images:
            glasses_prompts = [p for p in text_prompts if "glasses" in p.lower()]

            if glasses_prompts:
                glasses_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in glasses_prompts])
                        for r in glasses_images
                    ]
                )
                no_glasses_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in glasses_prompts])
                        for r in no_glasses_images
                    ]
                )
                print(
                    f"Images with glasses - 'glasses' prompts confidence: {glasses_conf:.3f}"
                )
                print(
                    f"Images without glasses - 'glasses' prompts confidence: {no_glasses_conf:.3f}"
                )

        receding_images = [
            r for r in results if "Receding_Hairline" in r["actual_attributes"]
        ]
        no_receding_images = [
            r for r in results if "Receding_Hairline" not in r["actual_attributes"]
        ]

        if receding_images and no_receding_images:
            receding_prompts = [
                p for p in text_prompts if "receding hairline" in p.lower()
            ]

            if receding_prompts:
                receding_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in receding_prompts])
                        for r in receding_images
                    ]
                )
                no_receding_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in receding_prompts])
                        for r in no_receding_images
                    ]
                )
                print(
                    f"Images with receding hairline - 'receding hairline' confidence: {receding_conf:.3f}"
                )
                print(
                    f"Images without receding hairline - 'receding hairline' confidence: {no_receding_conf:.3f}"
                )

        # attractive is performing quite bad subjective? both are ~0.13
        attractive_images = [
            r for r in results if "Attractive" in r["actual_attributes"]
        ]
        not_attractive_images = [
            r for r in results if "Attractive" not in r["actual_attributes"]
        ]

        if attractive_images and not_attractive_images:
            attractive_prompts = [p for p in text_prompts if "attractive" in p.lower()]

            if attractive_prompts:
                attractive_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in attractive_prompts])
                        for r in attractive_images
                    ]
                )
                not_attractive_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in attractive_prompts])
                        for r in not_attractive_images
                    ]
                )
                print(
                    f"Attractive images - 'attractive' prompts confidence: {attractive_conf:.3f}"
                )
                print(
                    f"Non-attractive images - 'attractive' prompts confidence: {not_attractive_conf:.3f}"
                )

        smiling_images = [r for r in results if "Smiling" in r["actual_attributes"]]
        not_smiling_images = [
            r for r in results if "Smiling" not in r["actual_attributes"]
        ]

        if smiling_images and not_smiling_images:
            smiling_prompts = [p for p in text_prompts if "smiling" in p.lower()]

            if smiling_prompts:
                smiling_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in smiling_prompts])
                        for r in smiling_images
                    ]
                )
                not_smiling_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in smiling_prompts])
                        for r in not_smiling_images
                    ]
                )
                print(
                    f"Smiling images - 'smiling' prompts confidence: {smiling_conf:.3f}"
                )
                print(
                    f"Non-smiling images - 'smiling' prompts confidence: {not_smiling_conf:.3f}"
                )

        makeup_images = [r for r in results if "Heavy_Makeup" in r["actual_attributes"]]
        no_makeup_images = [
            r for r in results if "Heavy_Makeup" not in r["actual_attributes"]
        ]

        if makeup_images and no_makeup_images:
            makeup_prompts = [p for p in text_prompts if "makeup" in p.lower()]

            if makeup_prompts:
                makeup_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in makeup_prompts])
                        for r in makeup_images
                    ]
                )
                no_makeup_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in makeup_prompts])
                        for r in no_makeup_images
                    ]
                )
                print(
                    f"Heavy makeup images - 'makeup' prompts confidence: {makeup_conf:.3f}"
                )
                print(
                    f"No heavy makeup images - 'makeup' prompts confidence: {no_makeup_conf:.3f}"
                )

        blonde_images = [r for r in results if "Blond_Hair" in r["actual_attributes"]]
        not_blonde_images = [
            r for r in results if "Blond_Hair" not in r["actual_attributes"]
        ]

        if blonde_images and not_blonde_images:
            blonde_prompts = [p for p in text_prompts if "blonde" in p.lower()]

            if blonde_prompts:
                blonde_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in blonde_prompts])
                        for r in blonde_images
                    ]
                )
                not_blonde_conf = np.mean(
                    [
                        np.mean([r["clip_predictions"][p] for p in blonde_prompts])
                        for r in not_blonde_images
                    ]
                )
                print(
                    f"Blonde hair images - 'blonde hair' confidence: {blonde_conf:.3f}"
                )
                print(
                    f"Non-blonde hair images - 'blonde hair' confidence: {not_blonde_conf:.3f}"
                )

        return results


def main():
    """Main function to run the complete analysis"""
    print("CelebA Dataset + CLIP Analysis")
    print("=" * 50)

    explorer = CelebAExplorer(root_dir="./data", download=True)

    # Explore dataset
    attr_df = explorer.explore_dataset(samples=10000)

    # show 8 samples in matplotlib
    explorer.visualize_sample_images()

    clip_results = explorer.run_clip_analysis(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        num_workers=DATALOADER_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    print("\nAnalysis complete!")
    print("Check 'celeba_samples.png' for sample images visualization")


if __name__ == "__main__":
    main()
