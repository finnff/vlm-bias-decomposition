# vlm-bias-decomposition

VLM bias mitigation via vector decomposition in CLIP

## Requirements

```bash
conda init #(if you haven't done this before)

conda create -n clipenv python=3.8 --yes
conda activate clipenv

conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm pandas matplotlib requests gdown scipy scikit-learn seaborn adjustText
pip install git+https://github.com/openai/CLIP.git
```


## Usage

1. First run `python celeba_downloader_clip_test.py` to ensure you have the CelebA dataset downloaded and CLIP model loaded.
2. Then run `python distinct_prompt_evaluation.py` to perform the CLIP Attribute Detection and Correlation Analysis.


### Nvidia CUDA

Repository is setup for usage with a CUDA which requires a Nvidia GPU, but will work on CPU as well (But a lot slower!). 
Eg. running both scripts with 10000 and 100000 samples respectively on a 8 year old Nvidia GTX 1080 with 8GB VRAM took less than 10 minutes in total.
The same scripts on my laptop(i7 8650u) would take over 4 hours with CPU processing, which is ~25x slower.
So please decrease the `NUM_SAMPLES` at the top of each script to get a more reasonable run time if you don't have a Nvidia GPU.

### celeba_downloader_clip_test.py



![img](./celeba_demo.png)

running `./celeba_downloader_clip_test.py` gives us: 
```
Using device: cuda
Loading CLIP model...
CelebA Dataset + CLIP Analysis
==================================================
Loading CelebA dataset...
Files already downloaded and verified
✓ Successfully loaded CelebA dataset with 162770 images

==================================================
CELEBA DATASET EXPLORATION
==================================================
Total number of images: 162770
Number of attributes: 40

Analyzing attributes distribution...
Sampling attributes: 100%|████████████████████████████████████████████████████████████████████████████| 162770/162770 [01:40<00:00, 1626.66it/s]

 Attributes and how frequent they are:
          Attribute    Count  Percentage
           No_Beard 135779.0   83.41
              Young 126788.0   77.89
         Attractive  83603.0   51.36
Mouth_Slightly_Open  78486.0   48.21
            Smiling  78080.0   47.96
   Wearing_Lipstick  76437.0   46.96
    High_Cheekbones  73645.0   45.24
               Male  68261.0   41.93
       Heavy_Makeup  62555.0   38.43
          Wavy_Hair  51982.0   31.93
          Oval_Face  46101.0   28.32
        Pointy_Nose  44846.0   27.55
    Arched_Eyebrows  43278.0   26.58
           Big_Lips  39213.0   24.09
         Black_Hair  38906.0   23.90
           Big_Nose  38341.0   23.55
      Straight_Hair  33947.0   20.85
    Bags_Under_Eyes  33280.0   20.44
         Brown_Hair  33192.0   20.39
   Wearing_Earrings  30362.0   18.65
              Bangs  24685.0   15.16
         Blond_Hair  24267.0   14.90
     Bushy_Eyebrows  23386.0   14.36
   Wearing_Necklace  19764.0   12.14
        Narrow_Eyes  18869.0   11.59
   5_o_Clock_Shadow  18177.0   11.16
  Receding_Hairline  13040.0    8.01
    Wearing_Necktie  11890.0    7.30
        Rosy_Cheeks  10525.0    6.46
         Eyeglasses  10521.0    6.46
             Goatee  10337.0    6.35
             Chubby   9389.0    5.76
          Sideburns   9156.0    5.62
             Blurry   8362.0    5.13
        Wearing_Hat   8039.0    4.93
        Double_Chin   7571.0    4.65
          Pale_Skin   7005.0    4.30
          Gray_Hair   6896.0    4.23
           Mustache   6642.0    4.08
               Bald   3713.0    2.28

Visualizing 8 sample images...

Running CLIP analysis on 162770 images...
Processing batches: 100%|█████████████████████████████████████████████████████████████████████████████████████| 636/636 [06:26<00:00,  1.64it/s]

Analyzing CLIP results...

Average CLIP confidences by text prompt:
a photo of a woman with makeup: 0.213
a photo of a woman smiling: 0.209
a photo of a man with a receding hairline: 0.198
a photo of a woman with blonde hair: 0.173
a photo of an attractive man: 0.130
a photo of a man with glasses: 0.077

Correlation analysis:
Male images - average 'man' prompts confidence: 0.167
Female images - average 'woman' prompts confidence: 0.330
Images with glasses - 'glasses' prompts confidence: 0.593
Images without glasses - 'glasses' prompts confidence: 0.042
Images with receding hairline - 'receding hairline' confidence: 0.431
Images without receding hairline - 'receding hairline' confidence: 0.177
Attractive images - 'attractive' prompts confidence: 0.106
Non-attractive images - 'attractive' prompts confidence: 0.154
Smiling images - 'smiling' prompts confidence: 0.347
Non-smiling images - 'smiling' prompts confidence: 0.081
Heavy makeup images - 'makeup' prompts confidence: 0.380
No heavy makeup images - 'makeup' prompts confidence: 0.109
Blonde hair images - 'blonde hair' confidence: 0.714
Non-blonde hair images - 'blonde hair' confidence: 0.079

Analysis complete!
Check 'celeba_samples.png' for sample images visualization
python3 celeba_downloader_clip_test.py  1568.52s user 95.02s system 332% cpu 8:21.06 total
```




## fixed_debias_clip_embeddings.py

This script performs a comprehensive analysis of gender bias in CLIP embeddings and the effectiveness of different debiasing techniques. It can be configured using the feature flags and parameters at the top of the file.

### Configuration

- `TOGGLE_GENDER_CLASSIFICATION`: (True/False) Enable/disable the gender classification analysis.
- `TOGGLE_PCA_VISUALIZATION`: (True/False) Enable/disable the PCA visualization of embeddings.
- `TOGGLE_TSNE_VISUALIZATION`: (True/False) Enable/disable the t-SNE visualization of embeddings.
- `TOGGLE_ATTRIBUTE_BIAS_DIRECTIONS`: (True/False) Enable/disable the analysis of attribute bias directions.
- `TOGGLE_MISCLASSIFIED_VISUALIZATION`: (True/False) Enable/disable the visualization of misclassified images.
- `TOGGLE_MALE_GROUP_COMPARISON`: (True/False) Enable/disable the t-test comparison of male groups.
- `TOGGLE_DEBIASING_ANALYSIS`: (True/False) Enable/disable the debiasing analysis.
- `MAX_SAMPLES`: (Integer) The number of samples to use from the dataset for the analysis.

### Usage

Run the script from the command line:

```bash
python fixed_debias_clip_embeddings.py
```

The script will perform the enabled analyses and save the results to the `result_imgs_100k` directory.
