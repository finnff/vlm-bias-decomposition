# vlm-bias-decomposition

VLM bias mitigation via vector decomposition in CLIP

## Requirements

```bash
conda init #(if you haven't done this before)

conda create -n clipenv python=3.8 --yes
conda activate clipenv

conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm pandas matplotlib requests gdown 
pip install git+https://github.com/openai/CLIP.git
```




![]
![img](./celeba_demo.png)

running `./celeba_downloader_clip_test.py` gives us: 
```
Using device: cuda
Loading CLIP model...
CelebA Dataset + CLIP Analysis
==================================================
Loading CelebA dataset...
✓ Successfully loaded CelebA dataset with 162770 images

==================================================
CELEBA DATASET EXPLORATION
==================================================
Total number of images: 162770
Number of attributes: 40

Analyzing attributes distribution...
Sampling attributes: 100%|█████████████████████████████████████████████████████████████████| 162770/162770 [01:25<00:00, 1898.20it/s]
\Attributes and how frequent they are:
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

Running CLIP analysis on 1000 images...
Processing images with CLIP: 100%|███████████████████████████████████████████████████████████████| 1000/1000 [01:06<00:00, 15.06it/s]

Analyzing CLIP results...

Average CLIP confidences by text prompt:
a photo of a woman with makeup: 0.212
a photo of a woman smiling: 0.203
a photo of a man with a receding hairline: 0.198
a photo of a woman with blonde hair: 0.169
a photo of an attractive man: 0.129
a photo of a man with glasses: 0.088

Correlation analysis:
Male images - average 'man' prompts confidence: 0.167
Female images - average 'woman' prompts confidence: 0.329
Images with glasses - 'glasses' prompts confidence: 0.719
Images without glasses - 'glasses' prompts confidence: 0.040
Images with receding hairline - 'receding hairline' confidence: 0.449
Images without receding hairline - 'receding hairline' confidence: 0.177
Attractive images - 'attractive' prompts confidence: 0.100
Non-attractive images - 'attractive' prompts confidence: 0.163
Smiling images - 'smiling' prompts confidence: 0.337
Non-smiling images - 'smiling' prompts confidence: 0.070
Heavy makeup images - 'makeup' prompts confidence: 0.383
No heavy makeup images - 'makeup' prompts confidence: 0.104
Blonde hair images - 'blonde hair' confidence: 0.727
Non-blonde hair images - 'blonde hair' confidence: 0.073

Analysis complete!
Check 'celeba_samples.png' for sample images visualization
```

_Note: Run time of this model for me with a 8g VRAM Nvidia card was 3 minutes, if it takes longer for you please lower the sample values at the bottom of the script!_ 
