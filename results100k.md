Starting CLIP Bias Analysis Pipeline
Output directory: result_imgs_100k
============================================================

Loading CelebA dataset...
Successfully loaded dataset with 162770 samples

Extracting CLIP embeddings for 100000 samples...
Getting labels: 100%|███████████████████████████████████████████████████████████████████████████| 196/196 [07:06<00:00,  2.17s/batch]
Getting all embeddings: 100%|███████████████████████████████████████████████████████████████████| 196/196 [06:47<00:00,  2.08s/batch]

============================================================
GENDER CLASSIFICATION ANALYSIS
============================================================
Gender classifier training accuracy: 0.9916
Gender classifier test accuracy:     0.9901
Classification Report:
              precision    recall  f1-score   support

      Female       0.99      0.99      0.99     11661
        Male       0.99      0.99      0.99      8410

    accuracy                           0.99     20071
   macro avg       0.99      0.99      0.99     20071
weighted avg       0.99      0.99      0.99     20071

Confusion Matrix:
[[11583    78]
 [  121  8289]]

Generating PCA visualization...
Saved plot: result_imgs_100k/gender_pca_custom.png

Generating t-SNE visualization...
Running t-SNE (this may take a moment)...
Saved plot: result_imgs_100k/gender_tsne_custom.png

Analyzing attribute bias directions...
Saved plot: result_imgs_100k/attribute_bias_all_with_arrows.png
Saved plot: result_imgs_100k/attribute_bias_selected_only.png

Attribute bias analysis (console output):
Top 10 attributes by CLIP‐embedding PCA separation:
   1. Male                  → distance = 3.6410
   2. Wearing_Lipstick      → distance = 3.1270
   3. Heavy_Makeup          → distance = 2.8514
   4. Bald                  → distance = 2.6150
   5. No_Beard              → distance = 2.5426
   6. Wearing_Necktie       → distance = 2.5308
   7. 5_o_Clock_Shadow      → distance = 2.4530
   8. Gray_Hair             → distance = 2.3473
   9. Sideburns             → distance = 2.3061
  10. Goatee                → distance = 2.2816
Top 5 Male‐aligned attributes (highest positive projection onto male axis):
   1. Male                  proj = 3.6410
   2. Bald                  proj = 2.5477
   3. Wearing_Necktie       proj = 2.5138
   4. Gray_Hair             proj = 2.3230
   5. 5_o_Clock_Shadow      proj = 2.3140

Top 5 Female‐aligned attributes (most negative projection onto male axis):
   1. Wearing_Lipstick      proj = -3.0763
   2. Heavy_Makeup          proj = -2.7551
   3. No_Beard              proj = -2.5254
   4. Attractive            proj = -1.9104
   5. Arched_Eyebrows       proj = -1.9026
Also saved original visualization to result_imgs_100k/attribute_bias_directions_original.png

Comparing male groups with different attributes...

============================================================
T-TEST ANALYSIS FOR INDIVIDUAL MALE ATTRIBUTES
============================================================
Applying Bonferroni correction. Significance level (alpha) = 1.282e-03

--- Attribute: 5_o_Clock_Shadow ---
Group 1 (Males with 5_o_Clock_Shadow): 11250 examples
Group 2 (Males without 5_o_Clock_Shadow): 30800 examples
Avg. P(male) for Group 1 (with 5_o_Clock_Shadow): 0.9973 ± 0.0268
Avg. P(male) for Group 2 (without 5_o_Clock_Shadow): 0.9783 ± 0.1137
SIGNIFICANT (p < 1.282e-03): t = 27.36, p = 3.309e-163

--- Attribute: Arched_Eyebrows ---
Group 1 (Males with Arched_Eyebrows): 2246 examples
Group 2 (Males without Arched_Eyebrows): 39804 examples
Avg. P(male) for Group 1 (with Arched_Eyebrows): 0.9797 ± 0.1089
Avg. P(male) for Group 2 (without Arched_Eyebrows): 0.9836 ± 0.0981
NOT SIGNIFICANT (p >= 1.282e-03): t = -1.68, p = 9.216e-02

--- Attribute: Attractive ---
Group 1 (Males with Attractive): 11700 examples
Group 2 (Males without Attractive): 30350 examples
Avg. P(male) for Group 1 (with Attractive): 0.9864 ± 0.0825
Avg. P(male) for Group 2 (without Attractive): 0.9822 ± 0.1042
SIGNIFICANT (p < 1.282e-03): t = 4.31, p = 1.625e-05

--- Attribute: Bags_Under_Eyes ---
Group 1 (Males with Bags_Under_Eyes): 14530 examples
Group 2 (Males without Bags_Under_Eyes): 27520 examples
Avg. P(male) for Group 1 (with Bags_Under_Eyes): 0.9892 ± 0.0782
Avg. P(male) for Group 2 (without Bags_Under_Eyes): 0.9803 ± 0.1078
SIGNIFICANT (p < 1.282e-03): t = 9.65, p = 5.019e-22

--- Attribute: Bald ---
Group 1 (Males with Bald): 2244 examples
Group 2 (Males without Bald): 39806 examples
Avg. P(male) for Group 1 (with Bald): 0.9977 ± 0.0325
Avg. P(male) for Group 2 (without Bald): 0.9826 ± 0.1011
SIGNIFICANT (p < 1.282e-03): t = 17.66, p = 7.218e-68

--- Attribute: Bangs ---
Group 1 (Males with Bangs): 3581 examples
Group 2 (Males without Bangs): 38469 examples
Avg. P(male) for Group 1 (with Bangs): 0.9571 ± 0.1489
Avg. P(male) for Group 2 (without Bangs): 0.9859 ± 0.0922
SIGNIFICANT (p < 1.282e-03): t = -11.35, p = 2.072e-29

--- Attribute: Big_Lips ---
Group 1 (Males with Big_Lips): 6427 examples
Group 2 (Males without Big_Lips): 35623 examples
Avg. P(male) for Group 1 (with Big_Lips): 0.9766 ± 0.1240
Avg. P(male) for Group 2 (without Big_Lips): 0.9846 ± 0.0933
SIGNIFICANT (p < 1.282e-03): t = -4.94, p = 7.887e-07

--- Attribute: Big_Nose ---
Group 1 (Males with Big_Nose): 17627 examples
Group 2 (Males without Big_Nose): 24423 examples
Avg. P(male) for Group 1 (with Big_Nose): 0.9892 ± 0.0796
Avg. P(male) for Group 2 (without Big_Nose): 0.9792 ± 0.1102
SIGNIFICANT (p < 1.282e-03): t = 10.75, p = 6.558e-27

--- Attribute: Black_Hair ---
Group 1 (Males with Black_Hair): 12473 examples
Group 2 (Males without Black_Hair): 29577 examples
Avg. P(male) for Group 1 (with Black_Hair): 0.9863 ± 0.0874
Avg. P(male) for Group 2 (without Black_Hair): 0.9822 ± 0.1030
SIGNIFICANT (p < 1.282e-03): t = 4.25, p = 2.127e-05

--- Attribute: Blond_Hair ---
Group 1 (Males with Blond_Hair): 857 examples
Group 2 (Males without Blond_Hair): 41193 examples
Avg. P(male) for Group 1 (with Blond_Hair): 0.9394 ± 0.1993
Avg. P(male) for Group 2 (without Blond_Hair): 0.9843 ± 0.0952
SIGNIFICANT (p < 1.282e-03): t = -6.58, p = 8.237e-11

--- Attribute: Blurry ---
Group 1 (Males with Blurry): 2460 examples
Group 2 (Males without Blurry): 39590 examples
Avg. P(male) for Group 1 (with Blurry): 0.9560 ± 0.1650
Avg. P(male) for Group 2 (without Blurry): 0.9851 ± 0.0927
SIGNIFICANT (p < 1.282e-03): t = -8.67, p = 7.489e-18

--- Attribute: Brown_Hair ---
Group 1 (Males with Brown_Hair): 6391 examples
Group 2 (Males without Brown_Hair): 35659 examples
Avg. P(male) for Group 1 (with Brown_Hair): 0.9828 ± 0.0973
Avg. P(male) for Group 2 (without Brown_Hair): 0.9835 ± 0.0989
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.50, p = 6.148e-01

--- Attribute: Bushy_Eyebrows ---
Group 1 (Males with Bushy_Eyebrows): 10267 examples
Group 2 (Males without Bushy_Eyebrows): 31783 examples
Avg. P(male) for Group 1 (with Bushy_Eyebrows): 0.9920 ± 0.0596
Avg. P(male) for Group 2 (without Bushy_Eyebrows): 0.9806 ± 0.1082
SIGNIFICANT (p < 1.282e-03): t = 13.43, p = 5.128e-41

--- Attribute: Chubby ---
Group 1 (Males with Chubby): 5034 examples
Group 2 (Males without Chubby): 37016 examples
Avg. P(male) for Group 1 (with Chubby): 0.9913 ± 0.0746
Avg. P(male) for Group 2 (without Chubby): 0.9823 ± 0.1015
SIGNIFICANT (p < 1.282e-03): t = 7.66, p = 2.161e-14

--- Attribute: Double_Chin ---
Group 1 (Males with Double_Chin): 4100 examples
Group 2 (Males without Double_Chin): 37950 examples
Avg. P(male) for Group 1 (with Double_Chin): 0.9944 ± 0.0578
Avg. P(male) for Group 2 (without Double_Chin): 0.9822 ± 0.1020
SIGNIFICANT (p < 1.282e-03): t = 11.63, p = 5.544e-31

--- Attribute: Eyeglasses ---
Group 1 (Males with Eyeglasses): 5181 examples
Group 2 (Males without Eyeglasses): 36869 examples
Avg. P(male) for Group 1 (with Eyeglasses): 0.9837 ± 0.1006
Avg. P(male) for Group 2 (without Eyeglasses): 0.9834 ± 0.0984
NOT SIGNIFICANT (p >= 1.282e-03): t = 0.27, p = 7.899e-01

--- Attribute: Goatee ---
Group 1 (Males with Goatee): 6272 examples
Group 2 (Males without Goatee): 35778 examples
Avg. P(male) for Group 1 (with Goatee): 0.9985 ± 0.0191
Avg. P(male) for Group 2 (without Goatee): 0.9807 ± 0.1065
SIGNIFICANT (p < 1.282e-03): t = 29.04, p = 1.275e-183

--- Attribute: Gray_Hair ---
Group 1 (Males with Gray_Hair): 3697 examples
Group 2 (Males without Gray_Hair): 38353 examples
Avg. P(male) for Group 1 (with Gray_Hair): 0.9920 ± 0.0715
Avg. P(male) for Group 2 (without Gray_Hair): 0.9826 ± 0.1009
SIGNIFICANT (p < 1.282e-03): t = 7.31, p = 2.978e-13

--- Attribute: Heavy_Makeup ---
Group 1 (Males with Heavy_Makeup): 122 examples
Group 2 (Males without Heavy_Makeup): 41928 examples
Avg. P(male) for Group 1 (with Heavy_Makeup): 0.8398 ± 0.2638
Avg. P(male) for Group 2 (without Heavy_Makeup): 0.9838 ± 0.0975
SIGNIFICANT (p < 1.282e-03): t = -6.01, p = 2.058e-08

--- Attribute: High_Cheekbones ---
Group 1 (Males with High_Cheekbones): 12885 examples
Group 2 (Males without High_Cheekbones): 29165 examples
Avg. P(male) for Group 1 (with High_Cheekbones): 0.9865 ± 0.0880
Avg. P(male) for Group 2 (without High_Cheekbones): 0.9820 ± 0.1030
SIGNIFICANT (p < 1.282e-03): t = 4.52, p = 6.241e-06

--- Attribute: Mouth_Slightly_Open ---
Group 1 (Males with Mouth_Slightly_Open): 17814 examples
Group 2 (Males without Mouth_Slightly_Open): 24236 examples
Avg. P(male) for Group 1 (with Mouth_Slightly_Open): 0.9834 ± 0.0978
Avg. P(male) for Group 2 (without Mouth_Slightly_Open): 0.9834 ± 0.0993
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.05, p = 9.603e-01

--- Attribute: Mustache ---
Group 1 (Males with Mustache): 4030 examples
Group 2 (Males without Mustache): 38020 examples
Avg. P(male) for Group 1 (with Mustache): 0.9986 ± 0.0176
Avg. P(male) for Group 2 (without Mustache): 0.9818 ± 0.1035
SIGNIFICANT (p < 1.282e-03): t = 28.01, p = 7.719e-171

--- Attribute: Narrow_Eyes ---
Group 1 (Males with Narrow_Eyes): 5068 examples
Group 2 (Males without Narrow_Eyes): 36982 examples
Avg. P(male) for Group 1 (with Narrow_Eyes): 0.9797 ± 0.1098
Avg. P(male) for Group 2 (without Narrow_Eyes): 0.9839 ± 0.0970
NOT SIGNIFICANT (p >= 1.282e-03): t = -2.59, p = 9.699e-03

--- Attribute: No_Beard ---
Group 1 (Males with No_Beard): 25535 examples
Group 2 (Males without No_Beard): 16515 examples
Avg. P(male) for Group 1 (with No_Beard): 0.9746 ± 0.1230
Avg. P(male) for Group 2 (without No_Beard): 0.9970 ± 0.0330
SIGNIFICANT (p < 1.282e-03): t = -27.57, p = 2.354e-165

--- Attribute: Oval_Face ---
Group 1 (Males with Oval_Face): 9241 examples
Group 2 (Males without Oval_Face): 32809 examples
Avg. P(male) for Group 1 (with Oval_Face): 0.9901 ± 0.0720
Avg. P(male) for Group 2 (without Oval_Face): 0.9815 ± 0.1049
SIGNIFICANT (p < 1.282e-03): t = 9.05, p = 1.490e-19

--- Attribute: Pale_Skin ---
Group 1 (Males with Pale_Skin): 1051 examples
Group 2 (Males without Pale_Skin): 40999 examples
Avg. P(male) for Group 1 (with Pale_Skin): 0.9542 ± 0.1676
Avg. P(male) for Group 2 (without Pale_Skin): 0.9841 ± 0.0962
SIGNIFICANT (p < 1.282e-03): t = -5.77, p = 1.065e-08

--- Attribute: Pointy_Nose ---
Group 1 (Males with Pointy_Nose): 6871 examples
Group 2 (Males without Pointy_Nose): 35179 examples
Avg. P(male) for Group 1 (with Pointy_Nose): 0.9873 ± 0.0863
Avg. P(male) for Group 2 (without Pointy_Nose): 0.9826 ± 0.1009
SIGNIFICANT (p < 1.282e-03): t = 3.96, p = 7.650e-05

--- Attribute: Receding_Hairline ---
Group 1 (Males with Receding_Hairline): 4914 examples
Group 2 (Males without Receding_Hairline): 37136 examples
Avg. P(male) for Group 1 (with Receding_Hairline): 0.9863 ± 0.1006
Avg. P(male) for Group 2 (without Receding_Hairline): 0.9830 ± 0.0984
NOT SIGNIFICANT (p >= 1.282e-03): t = 2.16, p = 3.094e-02

--- Attribute: Rosy_Cheeks ---
Group 1 (Males with Rosy_Cheeks): 132 examples
Group 2 (Males without Rosy_Cheeks): 41918 examples
Avg. P(male) for Group 1 (with Rosy_Cheeks): 0.9789 ± 0.1065
Avg. P(male) for Group 2 (without Rosy_Cheeks): 0.9834 ± 0.0987
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.49, p = 6.279e-01

--- Attribute: Sideburns ---
Group 1 (Males with Sideburns): 5612 examples
Group 2 (Males without Sideburns): 36438 examples
Avg. P(male) for Group 1 (with Sideburns): 0.9987 ± 0.0164
Avg. P(male) for Group 2 (without Sideburns): 0.9810 ± 0.1056
SIGNIFICANT (p < 1.282e-03): t = 29.76, p = 1.268e-192

--- Attribute: Smiling ---
Group 1 (Males with Smiling): 16746 examples
Group 2 (Males without Smiling): 25304 examples
Avg. P(male) for Group 1 (with Smiling): 0.9877 ± 0.0827
Avg. P(male) for Group 2 (without Smiling): 0.9805 ± 0.1079
SIGNIFICANT (p < 1.282e-03): t = 7.72, p = 1.184e-14

--- Attribute: Straight_Hair ---
Group 1 (Males with Straight_Hair): 10134 examples
Group 2 (Males without Straight_Hair): 31916 examples
Avg. P(male) for Group 1 (with Straight_Hair): 0.9869 ± 0.0844
Avg. P(male) for Group 2 (without Straight_Hair): 0.9823 ± 0.1028
SIGNIFICANT (p < 1.282e-03): t = 4.49, p = 7.097e-06

--- Attribute: Wavy_Hair ---
Group 1 (Males with Wavy_Hair): 6018 examples
Group 2 (Males without Wavy_Hair): 36032 examples
Avg. P(male) for Group 1 (with Wavy_Hair): 0.9752 ± 0.1189
Avg. P(male) for Group 2 (without Wavy_Hair): 0.9848 ± 0.0948
SIGNIFICANT (p < 1.282e-03): t = -5.94, p = 3.029e-09

--- Attribute: Wearing_Earrings ---
Group 1 (Males with Wearing_Earrings): 650 examples
Group 2 (Males without Wearing_Earrings): 41400 examples
Avg. P(male) for Group 1 (with Wearing_Earrings): 0.9289 ± 0.2250
Avg. P(male) for Group 2 (without Wearing_Earrings): 0.9843 ± 0.0951
SIGNIFICANT (p < 1.282e-03): t = -6.26, p = 6.799e-10

--- Attribute: Wearing_Hat ---
Group 1 (Males with Wearing_Hat): 3436 examples
Group 2 (Males without Wearing_Hat): 38614 examples
Avg. P(male) for Group 1 (with Wearing_Hat): 0.9760 ± 0.1171
Avg. P(male) for Group 2 (without Wearing_Hat): 0.9841 ± 0.0968
SIGNIFICANT (p < 1.282e-03): t = -3.90, p = 9.804e-05

--- Attribute: Wearing_Lipstick ---
Group 1 (Males with Wearing_Lipstick): 264 examples
Group 2 (Males without Wearing_Lipstick): 41786 examples
Avg. P(male) for Group 1 (with Wearing_Lipstick): 0.8577 ± 0.2619
Avg. P(male) for Group 2 (without Wearing_Lipstick): 0.9842 ± 0.0963
SIGNIFICANT (p < 1.282e-03): t = -7.83, p = 1.190e-13

--- Attribute: Wearing_Necklace ---
Group 1 (Males with Wearing_Necklace): 742 examples
Group 2 (Males without Wearing_Necklace): 41308 examples
Avg. P(male) for Group 1 (with Wearing_Necklace): 0.9474 ± 0.1691
Avg. P(male) for Group 2 (without Wearing_Necklace): 0.9840 ± 0.0968
SIGNIFICANT (p < 1.282e-03): t = -5.88, p = 6.156e-09

--- Attribute: Wearing_Necktie ---
Group 1 (Males with Wearing_Necktie): 7324 examples
Group 2 (Males without Wearing_Necktie): 34726 examples
Avg. P(male) for Group 1 (with Wearing_Necktie): 0.9974 ± 0.0294
Avg. P(male) for Group 2 (without Wearing_Necktie): 0.9804 ± 0.1075
SIGNIFICANT (p < 1.282e-03): t = 25.25, p = 1.317e-139

--- Attribute: Young ---
Group 1 (Males with Young): 26843 examples
Group 2 (Males without Young): 15207 examples
Avg. P(male) for Group 1 (with Young): 0.9810 ± 0.1041
Avg. P(male) for Group 2 (without Young): 0.9877 ± 0.0881
SIGNIFICANT (p < 1.282e-03): t = -6.98, p = 2.972e-12

============================================================

Performing debiasing analysis...

============================================================
DEBIASING ANALYSIS
============================================================

Original Embeddings:
Gender classifier training accuracy: 0.9916
Gender classifier test accuracy:     0.9901

============================================================
T-TEST ANALYSIS - Original
============================================================
Applying Bonferroni correction. Significance level (alpha) = 1.282e-03

--- Attribute: 5_o_Clock_Shadow ---
Group 1 (Males with 5_o_Clock_Shadow): 11250 examples
Group 2 (Males without 5_o_Clock_Shadow): 30800 examples
Avg. P(male) for Group 1 (with 5_o_Clock_Shadow): 0.9973 ± 0.0268
Avg. P(male) for Group 2 (without 5_o_Clock_Shadow): 0.9783 ± 0.1137
SIGNIFICANT (p < 1.282e-03): t = 27.36, p = 3.309e-163

--- Attribute: Arched_Eyebrows ---
Group 1 (Males with Arched_Eyebrows): 2246 examples
Group 2 (Males without Arched_Eyebrows): 39804 examples
Avg. P(male) for Group 1 (with Arched_Eyebrows): 0.9797 ± 0.1089
Avg. P(male) for Group 2 (without Arched_Eyebrows): 0.9836 ± 0.0981
NOT SIGNIFICANT (p >= 1.282e-03): t = -1.68, p = 9.216e-02

--- Attribute: Attractive ---
Group 1 (Males with Attractive): 11700 examples
Group 2 (Males without Attractive): 30350 examples
Avg. P(male) for Group 1 (with Attractive): 0.9864 ± 0.0825
Avg. P(male) for Group 2 (without Attractive): 0.9822 ± 0.1042
SIGNIFICANT (p < 1.282e-03): t = 4.31, p = 1.625e-05

--- Attribute: Bags_Under_Eyes ---
Group 1 (Males with Bags_Under_Eyes): 14530 examples
Group 2 (Males without Bags_Under_Eyes): 27520 examples
Avg. P(male) for Group 1 (with Bags_Under_Eyes): 0.9892 ± 0.0782
Avg. P(male) for Group 2 (without Bags_Under_Eyes): 0.9803 ± 0.1078
SIGNIFICANT (p < 1.282e-03): t = 9.65, p = 5.019e-22

--- Attribute: Bald ---
Group 1 (Males with Bald): 2244 examples
Group 2 (Males without Bald): 39806 examples
Avg. P(male) for Group 1 (with Bald): 0.9977 ± 0.0325
Avg. P(male) for Group 2 (without Bald): 0.9826 ± 0.1011
SIGNIFICANT (p < 1.282e-03): t = 17.66, p = 7.218e-68

--- Attribute: Bangs ---
Group 1 (Males with Bangs): 3581 examples
Group 2 (Males without Bangs): 38469 examples
Avg. P(male) for Group 1 (with Bangs): 0.9571 ± 0.1489
Avg. P(male) for Group 2 (without Bangs): 0.9859 ± 0.0922
SIGNIFICANT (p < 1.282e-03): t = -11.35, p = 2.072e-29

--- Attribute: Big_Lips ---
Group 1 (Males with Big_Lips): 6427 examples
Group 2 (Males without Big_Lips): 35623 examples
Avg. P(male) for Group 1 (with Big_Lips): 0.9766 ± 0.1240
Avg. P(male) for Group 2 (without Big_Lips): 0.9846 ± 0.0933
SIGNIFICANT (p < 1.282e-03): t = -4.94, p = 7.887e-07

--- Attribute: Big_Nose ---
Group 1 (Males with Big_Nose): 17627 examples
Group 2 (Males without Big_Nose): 24423 examples
Avg. P(male) for Group 1 (with Big_Nose): 0.9892 ± 0.0796
Avg. P(male) for Group 2 (without Big_Nose): 0.9792 ± 0.1102
SIGNIFICANT (p < 1.282e-03): t = 10.75, p = 6.558e-27

--- Attribute: Black_Hair ---
Group 1 (Males with Black_Hair): 12473 examples
Group 2 (Males without Black_Hair): 29577 examples
Avg. P(male) for Group 1 (with Black_Hair): 0.9863 ± 0.0874
Avg. P(male) for Group 2 (without Black_Hair): 0.9822 ± 0.1030
SIGNIFICANT (p < 1.282e-03): t = 4.25, p = 2.127e-05

--- Attribute: Blond_Hair ---
Group 1 (Males with Blond_Hair): 857 examples
Group 2 (Males without Blond_Hair): 41193 examples
Avg. P(male) for Group 1 (with Blond_Hair): 0.9394 ± 0.1993
Avg. P(male) for Group 2 (without Blond_Hair): 0.9843 ± 0.0952
SIGNIFICANT (p < 1.282e-03): t = -6.58, p = 8.237e-11

--- Attribute: Blurry ---
Group 1 (Males with Blurry): 2460 examples
Group 2 (Males without Blurry): 39590 examples
Avg. P(male) for Group 1 (with Blurry): 0.9560 ± 0.1650
Avg. P(male) for Group 2 (without Blurry): 0.9851 ± 0.0927
SIGNIFICANT (p < 1.282e-03): t = -8.67, p = 7.489e-18

--- Attribute: Brown_Hair ---
Group 1 (Males with Brown_Hair): 6391 examples
Group 2 (Males without Brown_Hair): 35659 examples
Avg. P(male) for Group 1 (with Brown_Hair): 0.9828 ± 0.0973
Avg. P(male) for Group 2 (without Brown_Hair): 0.9835 ± 0.0989
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.50, p = 6.148e-01

--- Attribute: Bushy_Eyebrows ---
Group 1 (Males with Bushy_Eyebrows): 10267 examples
Group 2 (Males without Bushy_Eyebrows): 31783 examples
Avg. P(male) for Group 1 (with Bushy_Eyebrows): 0.9920 ± 0.0596
Avg. P(male) for Group 2 (without Bushy_Eyebrows): 0.9806 ± 0.1082
SIGNIFICANT (p < 1.282e-03): t = 13.43, p = 5.128e-41

--- Attribute: Chubby ---
Group 1 (Males with Chubby): 5034 examples
Group 2 (Males without Chubby): 37016 examples
Avg. P(male) for Group 1 (with Chubby): 0.9913 ± 0.0746
Avg. P(male) for Group 2 (without Chubby): 0.9823 ± 0.1015
SIGNIFICANT (p < 1.282e-03): t = 7.66, p = 2.161e-14

--- Attribute: Double_Chin ---
Group 1 (Males with Double_Chin): 4100 examples
Group 2 (Males without Double_Chin): 37950 examples
Avg. P(male) for Group 1 (with Double_Chin): 0.9944 ± 0.0578
Avg. P(male) for Group 2 (without Double_Chin): 0.9822 ± 0.1020
SIGNIFICANT (p < 1.282e-03): t = 11.63, p = 5.544e-31

--- Attribute: Eyeglasses ---
Group 1 (Males with Eyeglasses): 5181 examples
Group 2 (Males without Eyeglasses): 36869 examples
Avg. P(male) for Group 1 (with Eyeglasses): 0.9837 ± 0.1006
Avg. P(male) for Group 2 (without Eyeglasses): 0.9834 ± 0.0984
NOT SIGNIFICANT (p >= 1.282e-03): t = 0.27, p = 7.899e-01

--- Attribute: Goatee ---
Group 1 (Males with Goatee): 6272 examples
Group 2 (Males without Goatee): 35778 examples
Avg. P(male) for Group 1 (with Goatee): 0.9985 ± 0.0191
Avg. P(male) for Group 2 (without Goatee): 0.9807 ± 0.1065
SIGNIFICANT (p < 1.282e-03): t = 29.04, p = 1.275e-183

--- Attribute: Gray_Hair ---
Group 1 (Males with Gray_Hair): 3697 examples
Group 2 (Males without Gray_Hair): 38353 examples
Avg. P(male) for Group 1 (with Gray_Hair): 0.9920 ± 0.0715
Avg. P(male) for Group 2 (without Gray_Hair): 0.9826 ± 0.1009
SIGNIFICANT (p < 1.282e-03): t = 7.31, p = 2.978e-13

--- Attribute: Heavy_Makeup ---
Group 1 (Males with Heavy_Makeup): 122 examples
Group 2 (Males without Heavy_Makeup): 41928 examples
Avg. P(male) for Group 1 (with Heavy_Makeup): 0.8398 ± 0.2638
Avg. P(male) for Group 2 (without Heavy_Makeup): 0.9838 ± 0.0975
SIGNIFICANT (p < 1.282e-03): t = -6.01, p = 2.058e-08

--- Attribute: High_Cheekbones ---
Group 1 (Males with High_Cheekbones): 12885 examples
Group 2 (Males without High_Cheekbones): 29165 examples
Avg. P(male) for Group 1 (with High_Cheekbones): 0.9865 ± 0.0880
Avg. P(male) for Group 2 (without High_Cheekbones): 0.9820 ± 0.1030
SIGNIFICANT (p < 1.282e-03): t = 4.52, p = 6.241e-06

--- Attribute: Mouth_Slightly_Open ---
Group 1 (Males with Mouth_Slightly_Open): 17814 examples
Group 2 (Males without Mouth_Slightly_Open): 24236 examples
Avg. P(male) for Group 1 (with Mouth_Slightly_Open): 0.9834 ± 0.0978
Avg. P(male) for Group 2 (without Mouth_Slightly_Open): 0.9834 ± 0.0993
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.05, p = 9.603e-01

--- Attribute: Mustache ---
Group 1 (Males with Mustache): 4030 examples
Group 2 (Males without Mustache): 38020 examples
Avg. P(male) for Group 1 (with Mustache): 0.9986 ± 0.0176
Avg. P(male) for Group 2 (without Mustache): 0.9818 ± 0.1035
SIGNIFICANT (p < 1.282e-03): t = 28.01, p = 7.719e-171

--- Attribute: Narrow_Eyes ---
Group 1 (Males with Narrow_Eyes): 5068 examples
Group 2 (Males without Narrow_Eyes): 36982 examples
Avg. P(male) for Group 1 (with Narrow_Eyes): 0.9797 ± 0.1098
Avg. P(male) for Group 2 (without Narrow_Eyes): 0.9839 ± 0.0970
NOT SIGNIFICANT (p >= 1.282e-03): t = -2.59, p = 9.699e-03

--- Attribute: No_Beard ---
Group 1 (Males with No_Beard): 25535 examples
Group 2 (Males without No_Beard): 16515 examples
Avg. P(male) for Group 1 (with No_Beard): 0.9746 ± 0.1230
Avg. P(male) for Group 2 (without No_Beard): 0.9970 ± 0.0330
SIGNIFICANT (p < 1.282e-03): t = -27.57, p = 2.354e-165

--- Attribute: Oval_Face ---
Group 1 (Males with Oval_Face): 9241 examples
Group 2 (Males without Oval_Face): 32809 examples
Avg. P(male) for Group 1 (with Oval_Face): 0.9901 ± 0.0720
Avg. P(male) for Group 2 (without Oval_Face): 0.9815 ± 0.1049
SIGNIFICANT (p < 1.282e-03): t = 9.05, p = 1.490e-19

--- Attribute: Pale_Skin ---
Group 1 (Males with Pale_Skin): 1051 examples
Group 2 (Males without Pale_Skin): 40999 examples
Avg. P(male) for Group 1 (with Pale_Skin): 0.9542 ± 0.1676
Avg. P(male) for Group 2 (without Pale_Skin): 0.9841 ± 0.0962
SIGNIFICANT (p < 1.282e-03): t = -5.77, p = 1.065e-08

--- Attribute: Pointy_Nose ---
Group 1 (Males with Pointy_Nose): 6871 examples
Group 2 (Males without Pointy_Nose): 35179 examples
Avg. P(male) for Group 1 (with Pointy_Nose): 0.9873 ± 0.0863
Avg. P(male) for Group 2 (without Pointy_Nose): 0.9826 ± 0.1009
SIGNIFICANT (p < 1.282e-03): t = 3.96, p = 7.650e-05

--- Attribute: Receding_Hairline ---
Group 1 (Males with Receding_Hairline): 4914 examples
Group 2 (Males without Receding_Hairline): 37136 examples
Avg. P(male) for Group 1 (with Receding_Hairline): 0.9863 ± 0.1006
Avg. P(male) for Group 2 (without Receding_Hairline): 0.9830 ± 0.0984
NOT SIGNIFICANT (p >= 1.282e-03): t = 2.16, p = 3.094e-02

--- Attribute: Rosy_Cheeks ---
Group 1 (Males with Rosy_Cheeks): 132 examples
Group 2 (Males without Rosy_Cheeks): 41918 examples
Avg. P(male) for Group 1 (with Rosy_Cheeks): 0.9789 ± 0.1065
Avg. P(male) for Group 2 (without Rosy_Cheeks): 0.9834 ± 0.0987
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.49, p = 6.279e-01

--- Attribute: Sideburns ---
Group 1 (Males with Sideburns): 5612 examples
Group 2 (Males without Sideburns): 36438 examples
Avg. P(male) for Group 1 (with Sideburns): 0.9987 ± 0.0164
Avg. P(male) for Group 2 (without Sideburns): 0.9810 ± 0.1056
SIGNIFICANT (p < 1.282e-03): t = 29.76, p = 1.268e-192

--- Attribute: Smiling ---
Group 1 (Males with Smiling): 16746 examples
Group 2 (Males without Smiling): 25304 examples
Avg. P(male) for Group 1 (with Smiling): 0.9877 ± 0.0827
Avg. P(male) for Group 2 (without Smiling): 0.9805 ± 0.1079
SIGNIFICANT (p < 1.282e-03): t = 7.72, p = 1.184e-14

--- Attribute: Straight_Hair ---
Group 1 (Males with Straight_Hair): 10134 examples
Group 2 (Males without Straight_Hair): 31916 examples
Avg. P(male) for Group 1 (with Straight_Hair): 0.9869 ± 0.0844
Avg. P(male) for Group 2 (without Straight_Hair): 0.9823 ± 0.1028
SIGNIFICANT (p < 1.282e-03): t = 4.49, p = 7.097e-06

--- Attribute: Wavy_Hair ---
Group 1 (Males with Wavy_Hair): 6018 examples
Group 2 (Males without Wavy_Hair): 36032 examples
Avg. P(male) for Group 1 (with Wavy_Hair): 0.9752 ± 0.1189
Avg. P(male) for Group 2 (without Wavy_Hair): 0.9848 ± 0.0948
SIGNIFICANT (p < 1.282e-03): t = -5.94, p = 3.029e-09

--- Attribute: Wearing_Earrings ---
Group 1 (Males with Wearing_Earrings): 650 examples
Group 2 (Males without Wearing_Earrings): 41400 examples
Avg. P(male) for Group 1 (with Wearing_Earrings): 0.9289 ± 0.2250
Avg. P(male) for Group 2 (without Wearing_Earrings): 0.9843 ± 0.0951
SIGNIFICANT (p < 1.282e-03): t = -6.26, p = 6.799e-10

--- Attribute: Wearing_Hat ---
Group 1 (Males with Wearing_Hat): 3436 examples
Group 2 (Males without Wearing_Hat): 38614 examples
Avg. P(male) for Group 1 (with Wearing_Hat): 0.9760 ± 0.1171
Avg. P(male) for Group 2 (without Wearing_Hat): 0.9841 ± 0.0968
SIGNIFICANT (p < 1.282e-03): t = -3.90, p = 9.804e-05

--- Attribute: Wearing_Lipstick ---
Group 1 (Males with Wearing_Lipstick): 264 examples
Group 2 (Males without Wearing_Lipstick): 41786 examples
Avg. P(male) for Group 1 (with Wearing_Lipstick): 0.8577 ± 0.2619
Avg. P(male) for Group 2 (without Wearing_Lipstick): 0.9842 ± 0.0963
SIGNIFICANT (p < 1.282e-03): t = -7.83, p = 1.190e-13

--- Attribute: Wearing_Necklace ---
Group 1 (Males with Wearing_Necklace): 742 examples
Group 2 (Males without Wearing_Necklace): 41308 examples
Avg. P(male) for Group 1 (with Wearing_Necklace): 0.9474 ± 0.1691
Avg. P(male) for Group 2 (without Wearing_Necklace): 0.9840 ± 0.0968
SIGNIFICANT (p < 1.282e-03): t = -5.88, p = 6.156e-09

--- Attribute: Wearing_Necktie ---
Group 1 (Males with Wearing_Necktie): 7324 examples
Group 2 (Males without Wearing_Necktie): 34726 examples
Avg. P(male) for Group 1 (with Wearing_Necktie): 0.9974 ± 0.0294
Avg. P(male) for Group 2 (without Wearing_Necktie): 0.9804 ± 0.1075
SIGNIFICANT (p < 1.282e-03): t = 25.25, p = 1.317e-139

--- Attribute: Young ---
Group 1 (Males with Young): 26843 examples
Group 2 (Males without Young): 15207 examples
Avg. P(male) for Group 1 (with Young): 0.9810 ± 0.1041
Avg. P(male) for Group 2 (without Young): 0.9877 ± 0.0881
SIGNIFICANT (p < 1.282e-03): t = -6.98, p = 2.972e-12

============================================================

Hard Debias Embeddings:
Gender classifier training accuracy: 0.6768
Gender classifier test accuracy:     0.6709

============================================================
T-TEST ANALYSIS - Hard Debias
============================================================
Applying Bonferroni correction. Significance level (alpha) = 1.282e-03

--- Attribute: 5_o_Clock_Shadow ---
Group 1 (Males with 5_o_Clock_Shadow): 11250 examples
Group 2 (Males without 5_o_Clock_Shadow): 30800 examples
Avg. P(male) for Group 1 (with 5_o_Clock_Shadow): 0.4038 ± 0.1605
Avg. P(male) for Group 2 (without 5_o_Clock_Shadow): 0.5365 ± 0.1678
SIGNIFICANT (p < 1.282e-03): t = -74.08, p = 0.000e+00

--- Attribute: Arched_Eyebrows ---
Group 1 (Males with Arched_Eyebrows): 2246 examples
Group 2 (Males without Arched_Eyebrows): 39804 examples
Avg. P(male) for Group 1 (with Arched_Eyebrows): 0.5345 ± 0.1708
Avg. P(male) for Group 2 (without Arched_Eyebrows): 0.4991 ± 0.1761
SIGNIFICANT (p < 1.282e-03): t = 9.54, p = 3.166e-21

--- Attribute: Attractive ---
Group 1 (Males with Attractive): 11700 examples
Group 2 (Males without Attractive): 30350 examples
Avg. P(male) for Group 1 (with Attractive): 0.4569 ± 0.1736
Avg. P(male) for Group 2 (without Attractive): 0.5180 ± 0.1739
SIGNIFICANT (p < 1.282e-03): t = -32.30, p = 1.954e-223

--- Attribute: Bags_Under_Eyes ---
Group 1 (Males with Bags_Under_Eyes): 14530 examples
Group 2 (Males without Bags_Under_Eyes): 27520 examples
Avg. P(male) for Group 1 (with Bags_Under_Eyes): 0.4945 ± 0.1714
Avg. P(male) for Group 2 (without Bags_Under_Eyes): 0.5044 ± 0.1782
SIGNIFICANT (p < 1.282e-03): t = -5.52, p = 3.460e-08

--- Attribute: Bald ---
Group 1 (Males with Bald): 2244 examples
Group 2 (Males without Bald): 39806 examples
Avg. P(male) for Group 1 (with Bald): 0.4023 ± 0.1440
Avg. P(male) for Group 2 (without Bald): 0.5065 ± 0.1760
SIGNIFICANT (p < 1.282e-03): t = -32.92, p = 2.370e-199

--- Attribute: Bangs ---
Group 1 (Males with Bangs): 3581 examples
Group 2 (Males without Bangs): 38469 examples
Avg. P(male) for Group 1 (with Bangs): 0.5802 ± 0.1624
Avg. P(male) for Group 2 (without Bangs): 0.4936 ± 0.1754
SIGNIFICANT (p < 1.282e-03): t = 30.29, p = 3.699e-183

--- Attribute: Big_Lips ---
Group 1 (Males with Big_Lips): 6427 examples
Group 2 (Males without Big_Lips): 35623 examples
Avg. P(male) for Group 1 (with Big_Lips): 0.5386 ± 0.1845
Avg. P(male) for Group 2 (without Big_Lips): 0.4942 ± 0.1735
SIGNIFICANT (p < 1.282e-03): t = 17.94, p = 1.081e-70

--- Attribute: Big_Nose ---
Group 1 (Males with Big_Nose): 17627 examples
Group 2 (Males without Big_Nose): 24423 examples
Avg. P(male) for Group 1 (with Big_Nose): 0.5152 ± 0.1719
Avg. P(male) for Group 2 (without Big_Nose): 0.4907 ± 0.1782
SIGNIFICANT (p < 1.282e-03): t = 14.19, p = 1.401e-45

--- Attribute: Black_Hair ---
Group 1 (Males with Black_Hair): 12473 examples
Group 2 (Males without Black_Hair): 29577 examples
Avg. P(male) for Group 1 (with Black_Hair): 0.5202 ± 0.1796
Avg. P(male) for Group 2 (without Black_Hair): 0.4929 ± 0.1738
SIGNIFICANT (p < 1.282e-03): t = 14.40, p = 8.827e-47

--- Attribute: Blond_Hair ---
Group 1 (Males with Blond_Hair): 857 examples
Group 2 (Males without Blond_Hair): 41193 examples
Avg. P(male) for Group 1 (with Blond_Hair): 0.5099 ± 0.1675
Avg. P(male) for Group 2 (without Blond_Hair): 0.5008 ± 0.1761
NOT SIGNIFICANT (p >= 1.282e-03): t = 1.58, p = 1.153e-01

--- Attribute: Blurry ---
Group 1 (Males with Blurry): 2460 examples
Group 2 (Males without Blurry): 39590 examples
Avg. P(male) for Group 1 (with Blurry): 0.5052 ± 0.1907
Avg. P(male) for Group 2 (without Blurry): 0.5007 ± 0.1750
NOT SIGNIFICANT (p >= 1.282e-03): t = 1.13, p = 2.592e-01

--- Attribute: Brown_Hair ---
Group 1 (Males with Brown_Hair): 6391 examples
Group 2 (Males without Brown_Hair): 35659 examples
Avg. P(male) for Group 1 (with Brown_Hair): 0.4892 ± 0.1730
Avg. P(male) for Group 2 (without Brown_Hair): 0.5031 ± 0.1764
SIGNIFICANT (p < 1.282e-03): t = -5.87, p = 4.531e-09

--- Attribute: Bushy_Eyebrows ---
Group 1 (Males with Bushy_Eyebrows): 10267 examples
Group 2 (Males without Bushy_Eyebrows): 31783 examples
Avg. P(male) for Group 1 (with Bushy_Eyebrows): 0.4842 ± 0.1761
Avg. P(male) for Group 2 (without Bushy_Eyebrows): 0.5064 ± 0.1756
SIGNIFICANT (p < 1.282e-03): t = -11.15, p = 9.447e-29

--- Attribute: Chubby ---
Group 1 (Males with Chubby): 5034 examples
Group 2 (Males without Chubby): 37016 examples
Avg. P(male) for Group 1 (with Chubby): 0.5416 ± 0.1691
Avg. P(male) for Group 2 (without Chubby): 0.4955 ± 0.1762
SIGNIFICANT (p < 1.282e-03): t = 18.07, p = 2.880e-71

--- Attribute: Double_Chin ---
Group 1 (Males with Double_Chin): 4100 examples
Group 2 (Males without Double_Chin): 37950 examples
Avg. P(male) for Group 1 (with Double_Chin): 0.5251 ± 0.1642
Avg. P(male) for Group 2 (without Double_Chin): 0.4984 ± 0.1770
SIGNIFICANT (p < 1.282e-03): t = 9.81, p = 1.673e-22

--- Attribute: Eyeglasses ---
Group 1 (Males with Eyeglasses): 5181 examples
Group 2 (Males without Eyeglasses): 36869 examples
Avg. P(male) for Group 1 (with Eyeglasses): 0.5005 ± 0.1767
Avg. P(male) for Group 2 (without Eyeglasses): 0.5011 ± 0.1759
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.22, p = 8.225e-01

--- Attribute: Goatee ---
Group 1 (Males with Goatee): 6272 examples
Group 2 (Males without Goatee): 35778 examples
Avg. P(male) for Group 1 (with Goatee): 0.4898 ± 0.1767
Avg. P(male) for Group 2 (without Goatee): 0.5029 ± 0.1758
SIGNIFICANT (p < 1.282e-03): t = -5.44, p = 5.329e-08

--- Attribute: Gray_Hair ---
Group 1 (Males with Gray_Hair): 3697 examples
Group 2 (Males without Gray_Hair): 38353 examples
Avg. P(male) for Group 1 (with Gray_Hair): 0.4350 ± 0.1462
Avg. P(male) for Group 2 (without Gray_Hair): 0.5073 ± 0.1773
SIGNIFICANT (p < 1.282e-03): t = -28.14, p = 1.934e-161

--- Attribute: Heavy_Makeup ---
Group 1 (Males with Heavy_Makeup): 122 examples
Group 2 (Males without Heavy_Makeup): 41928 examples
Avg. P(male) for Group 1 (with Heavy_Makeup): 0.6123 ± 0.1527
Avg. P(male) for Group 2 (without Heavy_Makeup): 0.5007 ± 0.1759
SIGNIFICANT (p < 1.282e-03): t = 8.02, p = 7.165e-13

--- Attribute: High_Cheekbones ---
Group 1 (Males with High_Cheekbones): 12885 examples
Group 2 (Males without High_Cheekbones): 29165 examples
Avg. P(male) for Group 1 (with High_Cheekbones): 0.5155 ± 0.1664
Avg. P(male) for Group 2 (without High_Cheekbones): 0.4946 ± 0.1796
SIGNIFICANT (p < 1.282e-03): t = 11.61, p = 4.331e-31

--- Attribute: Mouth_Slightly_Open ---
Group 1 (Males with Mouth_Slightly_Open): 17814 examples
Group 2 (Males without Mouth_Slightly_Open): 24236 examples
Avg. P(male) for Group 1 (with Mouth_Slightly_Open): 0.5096 ± 0.1731
Avg. P(male) for Group 2 (without Mouth_Slightly_Open): 0.4946 ± 0.1778
SIGNIFICANT (p < 1.282e-03): t = 8.68, p = 4.067e-18

--- Attribute: Mustache ---
Group 1 (Males with Mustache): 4030 examples
Group 2 (Males without Mustache): 38020 examples
Avg. P(male) for Group 1 (with Mustache): 0.5106 ± 0.1858
Avg. P(male) for Group 2 (without Mustache): 0.5000 ± 0.1749
SIGNIFICANT (p < 1.282e-03): t = 3.49, p = 4.848e-04

--- Attribute: Narrow_Eyes ---
Group 1 (Males with Narrow_Eyes): 5068 examples
Group 2 (Males without Narrow_Eyes): 36982 examples
Avg. P(male) for Group 1 (with Narrow_Eyes): 0.4963 ± 0.1803
Avg. P(male) for Group 2 (without Narrow_Eyes): 0.5016 ± 0.1754
NOT SIGNIFICANT (p >= 1.282e-03): t = -1.99, p = 4.664e-02

--- Attribute: No_Beard ---
Group 1 (Males with No_Beard): 25535 examples
Group 2 (Males without No_Beard): 16515 examples
Avg. P(male) for Group 1 (with No_Beard): 0.5303 ± 0.1659
Avg. P(male) for Group 2 (without No_Beard): 0.4556 ± 0.1813
SIGNIFICANT (p < 1.282e-03): t = 42.68, p = 0.000e+00

--- Attribute: Oval_Face ---
Group 1 (Males with Oval_Face): 9241 examples
Group 2 (Males without Oval_Face): 32809 examples
Avg. P(male) for Group 1 (with Oval_Face): 0.5087 ± 0.1669
Avg. P(male) for Group 2 (without Oval_Face): 0.4988 ± 0.1784
SIGNIFICANT (p < 1.282e-03): t = 4.93, p = 8.297e-07

--- Attribute: Pale_Skin ---
Group 1 (Males with Pale_Skin): 1051 examples
Group 2 (Males without Pale_Skin): 40999 examples
Avg. P(male) for Group 1 (with Pale_Skin): 0.5265 ± 0.1768
Avg. P(male) for Group 2 (without Pale_Skin): 0.5003 ± 0.1759
SIGNIFICANT (p < 1.282e-03): t = 4.75, p = 2.342e-06

--- Attribute: Pointy_Nose ---
Group 1 (Males with Pointy_Nose): 6871 examples
Group 2 (Males without Pointy_Nose): 35179 examples
Avg. P(male) for Group 1 (with Pointy_Nose): 0.4536 ± 0.1731
Avg. P(male) for Group 2 (without Pointy_Nose): 0.5102 ± 0.1750
SIGNIFICANT (p < 1.282e-03): t = -24.75, p = 2.942e-131

--- Attribute: Receding_Hairline ---
Group 1 (Males with Receding_Hairline): 4914 examples
Group 2 (Males without Receding_Hairline): 37136 examples
Avg. P(male) for Group 1 (with Receding_Hairline): 0.4994 ± 0.1675
Avg. P(male) for Group 2 (without Receding_Hairline): 0.5012 ± 0.1770
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.71, p = 4.756e-01

--- Attribute: Rosy_Cheeks ---
Group 1 (Males with Rosy_Cheeks): 132 examples
Group 2 (Males without Rosy_Cheeks): 41918 examples
Avg. P(male) for Group 1 (with Rosy_Cheeks): 0.4463 ± 0.1592
Avg. P(male) for Group 2 (without Rosy_Cheeks): 0.5012 ± 0.1760
SIGNIFICANT (p < 1.282e-03): t = -3.93, p = 1.354e-04

--- Attribute: Sideburns ---
Group 1 (Males with Sideburns): 5612 examples
Group 2 (Males without Sideburns): 36438 examples
Avg. P(male) for Group 1 (with Sideburns): 0.4271 ± 0.1767
Avg. P(male) for Group 2 (without Sideburns): 0.5124 ± 0.1731
SIGNIFICANT (p < 1.282e-03): t = -33.72, p = 5.813e-232

--- Attribute: Smiling ---
Group 1 (Males with Smiling): 16746 examples
Group 2 (Males without Smiling): 25304 examples
Avg. P(male) for Group 1 (with Smiling): 0.5022 ± 0.1685
Avg. P(male) for Group 2 (without Smiling): 0.5002 ± 0.1807
NOT SIGNIFICANT (p >= 1.282e-03): t = 1.14, p = 2.524e-01

--- Attribute: Straight_Hair ---
Group 1 (Males with Straight_Hair): 10134 examples
Group 2 (Males without Straight_Hair): 31916 examples
Avg. P(male) for Group 1 (with Straight_Hair): 0.5005 ± 0.1726
Avg. P(male) for Group 2 (without Straight_Hair): 0.5011 ± 0.1770
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.32, p = 7.508e-01

--- Attribute: Wavy_Hair ---
Group 1 (Males with Wavy_Hair): 6018 examples
Group 2 (Males without Wavy_Hair): 36032 examples
Avg. P(male) for Group 1 (with Wavy_Hair): 0.5093 ± 0.1782
Avg. P(male) for Group 2 (without Wavy_Hair): 0.4996 ± 0.1755
SIGNIFICANT (p < 1.282e-03): t = 3.94, p = 8.371e-05

--- Attribute: Wearing_Earrings ---
Group 1 (Males with Wearing_Earrings): 650 examples
Group 2 (Males without Wearing_Earrings): 41400 examples
Avg. P(male) for Group 1 (with Wearing_Earrings): 0.5750 ± 0.1781
Avg. P(male) for Group 2 (without Wearing_Earrings): 0.4998 ± 0.1757
SIGNIFICANT (p < 1.282e-03): t = 10.67, p = 1.185e-24

--- Attribute: Wearing_Hat ---
Group 1 (Males with Wearing_Hat): 3436 examples
Group 2 (Males without Wearing_Hat): 38614 examples
Avg. P(male) for Group 1 (with Wearing_Hat): 0.5300 ± 0.1841
Avg. P(male) for Group 2 (without Wearing_Hat): 0.4984 ± 0.1750
SIGNIFICANT (p < 1.282e-03): t = 9.67, p = 6.830e-22

--- Attribute: Wearing_Lipstick ---
Group 1 (Males with Wearing_Lipstick): 264 examples
Group 2 (Males without Wearing_Lipstick): 41786 examples
Avg. P(male) for Group 1 (with Wearing_Lipstick): 0.6164 ± 0.1619
Avg. P(male) for Group 2 (without Wearing_Lipstick): 0.5003 ± 0.1758
SIGNIFICANT (p < 1.282e-03): t = 11.59, p = 2.012e-25

--- Attribute: Wearing_Necklace ---
Group 1 (Males with Wearing_Necklace): 742 examples
Group 2 (Males without Wearing_Necklace): 41308 examples
Avg. P(male) for Group 1 (with Wearing_Necklace): 0.5542 ± 0.1874
Avg. P(male) for Group 2 (without Wearing_Necklace): 0.5000 ± 0.1756
SIGNIFICANT (p < 1.282e-03): t = 7.82, p = 1.812e-14

--- Attribute: Wearing_Necktie ---
Group 1 (Males with Wearing_Necktie): 7324 examples
Group 2 (Males without Wearing_Necktie): 34726 examples
Avg. P(male) for Group 1 (with Wearing_Necktie): 0.4990 ± 0.1672
Avg. P(male) for Group 2 (without Wearing_Necktie): 0.5014 ± 0.1778
NOT SIGNIFICANT (p >= 1.282e-03): t = -1.09, p = 2.743e-01

--- Attribute: Young ---
Group 1 (Males with Young): 26843 examples
Group 2 (Males without Young): 15207 examples
Avg. P(male) for Group 1 (with Young): 0.4901 ± 0.1791
Avg. P(male) for Group 2 (without Young): 0.5202 ± 0.1686
SIGNIFICANT (p < 1.282e-03): t = -17.22, p = 3.781e-66

============================================================

Soft Debias(λ=3.5) Embeddings:
Gender classifier training accuracy: 0.9916
Gender classifier test accuracy:     0.9902

============================================================
T-TEST ANALYSIS - Soft Debias(λ=3.5)
============================================================
Applying Bonferroni correction. Significance level (alpha) = 1.282e-03

--- Attribute: 5_o_Clock_Shadow ---
Group 1 (Males with 5_o_Clock_Shadow): 11250 examples
Group 2 (Males without 5_o_Clock_Shadow): 30800 examples
Avg. P(male) for Group 1 (with 5_o_Clock_Shadow): 0.9966 ± 0.0269
Avg. P(male) for Group 2 (without 5_o_Clock_Shadow): 0.9776 ± 0.1138
SIGNIFICANT (p < 1.282e-03): t = 27.26, p = 4.550e-162

--- Attribute: Arched_Eyebrows ---
Group 1 (Males with Arched_Eyebrows): 2246 examples
Group 2 (Males without Arched_Eyebrows): 39804 examples
Avg. P(male) for Group 1 (with Arched_Eyebrows): 0.9791 ± 0.1087
Avg. P(male) for Group 2 (without Arched_Eyebrows): 0.9829 ± 0.0981
NOT SIGNIFICANT (p >= 1.282e-03): t = -1.65, p = 1.000e-01

--- Attribute: Attractive ---
Group 1 (Males with Attractive): 11700 examples
Group 2 (Males without Attractive): 30350 examples
Avg. P(male) for Group 1 (with Attractive): 0.9857 ± 0.0822
Avg. P(male) for Group 2 (without Attractive): 0.9815 ± 0.1044
SIGNIFICANT (p < 1.282e-03): t = 4.32, p = 1.559e-05

--- Attribute: Bags_Under_Eyes ---
Group 1 (Males with Bags_Under_Eyes): 14530 examples
Group 2 (Males without Bags_Under_Eyes): 27520 examples
Avg. P(male) for Group 1 (with Bags_Under_Eyes): 0.9885 ± 0.0783
Avg. P(male) for Group 2 (without Bags_Under_Eyes): 0.9796 ± 0.1079
SIGNIFICANT (p < 1.282e-03): t = 9.71, p = 2.809e-22

--- Attribute: Bald ---
Group 1 (Males with Bald): 2244 examples
Group 2 (Males without Bald): 39806 examples
Avg. P(male) for Group 1 (with Bald): 0.9969 ± 0.0335
Avg. P(male) for Group 2 (without Bald): 0.9819 ± 0.1011
SIGNIFICANT (p < 1.282e-03): t = 17.29, p = 4.190e-65

--- Attribute: Bangs ---
Group 1 (Males with Bangs): 3581 examples
Group 2 (Males without Bangs): 38469 examples
Avg. P(male) for Group 1 (with Bangs): 0.9567 ± 0.1480
Avg. P(male) for Group 2 (without Bangs): 0.9851 ± 0.0925
SIGNIFICANT (p < 1.282e-03): t = -11.28, p = 4.545e-29

--- Attribute: Big_Lips ---
Group 1 (Males with Big_Lips): 6427 examples
Group 2 (Males without Big_Lips): 35623 examples
Avg. P(male) for Group 1 (with Big_Lips): 0.9759 ± 0.1244
Avg. P(male) for Group 2 (without Big_Lips): 0.9839 ± 0.0933
SIGNIFICANT (p < 1.282e-03): t = -4.94, p = 7.836e-07

--- Attribute: Big_Nose ---
Group 1 (Males with Big_Nose): 17627 examples
Group 2 (Males without Big_Nose): 24423 examples
Avg. P(male) for Group 1 (with Big_Nose): 0.9885 ± 0.0798
Avg. P(male) for Group 2 (without Big_Nose): 0.9785 ± 0.1102
SIGNIFICANT (p < 1.282e-03): t = 10.83, p = 2.771e-27

--- Attribute: Black_Hair ---
Group 1 (Males with Black_Hair): 12473 examples
Group 2 (Males without Black_Hair): 29577 examples
Avg. P(male) for Group 1 (with Black_Hair): 0.9857 ± 0.0875
Avg. P(male) for Group 2 (without Black_Hair): 0.9814 ± 0.1031
SIGNIFICANT (p < 1.282e-03): t = 4.34, p = 1.408e-05

--- Attribute: Blond_Hair ---
Group 1 (Males with Blond_Hair): 857 examples
Group 2 (Males without Blond_Hair): 41193 examples
Avg. P(male) for Group 1 (with Blond_Hair): 0.9385 ± 0.1984
Avg. P(male) for Group 2 (without Blond_Hair): 0.9836 ± 0.0953
SIGNIFICANT (p < 1.282e-03): t = -6.64, p = 5.662e-11

--- Attribute: Blurry ---
Group 1 (Males with Blurry): 2460 examples
Group 2 (Males without Blurry): 39590 examples
Avg. P(male) for Group 1 (with Blurry): 0.9551 ± 0.1650
Avg. P(male) for Group 2 (without Blurry): 0.9844 ± 0.0928
SIGNIFICANT (p < 1.282e-03): t = -8.74, p = 4.288e-18

--- Attribute: Brown_Hair ---
Group 1 (Males with Brown_Hair): 6391 examples
Group 2 (Males without Brown_Hair): 35659 examples
Avg. P(male) for Group 1 (with Brown_Hair): 0.9821 ± 0.0973
Avg. P(male) for Group 2 (without Brown_Hair): 0.9828 ± 0.0990
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.58, p = 5.632e-01

--- Attribute: Bushy_Eyebrows ---
Group 1 (Males with Bushy_Eyebrows): 10267 examples
Group 2 (Males without Bushy_Eyebrows): 31783 examples
Avg. P(male) for Group 1 (with Bushy_Eyebrows): 0.9912 ± 0.0597
Avg. P(male) for Group 2 (without Bushy_Eyebrows): 0.9799 ± 0.1082
SIGNIFICANT (p < 1.282e-03): t = 13.36, p = 1.300e-40

--- Attribute: Chubby ---
Group 1 (Males with Chubby): 5034 examples
Group 2 (Males without Chubby): 37016 examples
Avg. P(male) for Group 1 (with Chubby): 0.9907 ± 0.0749
Avg. P(male) for Group 2 (without Chubby): 0.9816 ± 0.1015
SIGNIFICANT (p < 1.282e-03): t = 7.72, p = 1.334e-14

--- Attribute: Double_Chin ---
Group 1 (Males with Double_Chin): 4100 examples
Group 2 (Males without Double_Chin): 37950 examples
Avg. P(male) for Group 1 (with Double_Chin): 0.9938 ± 0.0579
Avg. P(male) for Group 2 (without Double_Chin): 0.9815 ± 0.1021
SIGNIFICANT (p < 1.282e-03): t = 11.77, p = 1.064e-31

--- Attribute: Eyeglasses ---
Group 1 (Males with Eyeglasses): 5181 examples
Group 2 (Males without Eyeglasses): 36869 examples
Avg. P(male) for Group 1 (with Eyeglasses): 0.9831 ± 0.1010
Avg. P(male) for Group 2 (without Eyeglasses): 0.9826 ± 0.0984
NOT SIGNIFICANT (p >= 1.282e-03): t = 0.33, p = 7.399e-01

--- Attribute: Goatee ---
Group 1 (Males with Goatee): 6272 examples
Group 2 (Males without Goatee): 35778 examples
Avg. P(male) for Group 1 (with Goatee): 0.9980 ± 0.0193
Avg. P(male) for Group 2 (without Goatee): 0.9800 ± 0.1065
SIGNIFICANT (p < 1.282e-03): t = 29.22, p = 7.070e-186

--- Attribute: Gray_Hair ---
Group 1 (Males with Gray_Hair): 3697 examples
Group 2 (Males without Gray_Hair): 38353 examples
Avg. P(male) for Group 1 (with Gray_Hair): 0.9913 ± 0.0723
Avg. P(male) for Group 2 (without Gray_Hair): 0.9819 ± 0.1009
SIGNIFICANT (p < 1.282e-03): t = 7.23, p = 5.358e-13

--- Attribute: Heavy_Makeup ---
Group 1 (Males with Heavy_Makeup): 122 examples
Group 2 (Males without Heavy_Makeup): 41928 examples
Avg. P(male) for Group 1 (with Heavy_Makeup): 0.8400 ± 0.2607
Avg. P(male) for Group 2 (without Heavy_Makeup): 0.9831 ± 0.0976
SIGNIFICANT (p < 1.282e-03): t = -6.04, p = 1.772e-08

--- Attribute: High_Cheekbones ---
Group 1 (Males with High_Cheekbones): 12885 examples
Group 2 (Males without High_Cheekbones): 29165 examples
Avg. P(male) for Group 1 (with High_Cheekbones): 0.9858 ± 0.0880
Avg. P(male) for Group 2 (without High_Cheekbones): 0.9813 ± 0.1031
SIGNIFICANT (p < 1.282e-03): t = 4.60, p = 4.320e-06

--- Attribute: Mouth_Slightly_Open ---
Group 1 (Males with Mouth_Slightly_Open): 17814 examples
Group 2 (Males without Mouth_Slightly_Open): 24236 examples
Avg. P(male) for Group 1 (with Mouth_Slightly_Open): 0.9828 ± 0.0974
Avg. P(male) for Group 2 (without Mouth_Slightly_Open): 0.9827 ± 0.0997
NOT SIGNIFICANT (p >= 1.282e-03): t = 0.11, p = 9.121e-01

--- Attribute: Mustache ---
Group 1 (Males with Mustache): 4030 examples
Group 2 (Males without Mustache): 38020 examples
Avg. P(male) for Group 1 (with Mustache): 0.9980 ± 0.0181
Avg. P(male) for Group 2 (without Mustache): 0.9811 ± 0.1035
SIGNIFICANT (p < 1.282e-03): t = 28.02, p = 6.006e-171

--- Attribute: Narrow_Eyes ---
Group 1 (Males with Narrow_Eyes): 5068 examples
Group 2 (Males without Narrow_Eyes): 36982 examples
Avg. P(male) for Group 1 (with Narrow_Eyes): 0.9790 ± 0.1098
Avg. P(male) for Group 2 (without Narrow_Eyes): 0.9832 ± 0.0971
NOT SIGNIFICANT (p >= 1.282e-03): t = -2.59, p = 9.707e-03

--- Attribute: No_Beard ---
Group 1 (Males with No_Beard): 25535 examples
Group 2 (Males without No_Beard): 16515 examples
Avg. P(male) for Group 1 (with No_Beard): 0.9739 ± 0.1230
Avg. P(male) for Group 2 (without No_Beard): 0.9963 ± 0.0333
SIGNIFICANT (p < 1.282e-03): t = -27.60, p = 1.052e-165

--- Attribute: Oval_Face ---
Group 1 (Males with Oval_Face): 9241 examples
Group 2 (Males without Oval_Face): 32809 examples
Avg. P(male) for Group 1 (with Oval_Face): 0.9895 ± 0.0721
Avg. P(male) for Group 2 (without Oval_Face): 0.9808 ± 0.1049
SIGNIFICANT (p < 1.282e-03): t = 9.16, p = 5.676e-20

--- Attribute: Pale_Skin ---
Group 1 (Males with Pale_Skin): 1051 examples
Group 2 (Males without Pale_Skin): 40999 examples
Avg. P(male) for Group 1 (with Pale_Skin): 0.9530 ± 0.1683
Avg. P(male) for Group 2 (without Pale_Skin): 0.9835 ± 0.0962
SIGNIFICANT (p < 1.282e-03): t = -5.83, p = 7.232e-09

--- Attribute: Pointy_Nose ---
Group 1 (Males with Pointy_Nose): 6871 examples
Group 2 (Males without Pointy_Nose): 35179 examples
Avg. P(male) for Group 1 (with Pointy_Nose): 0.9865 ± 0.0866
Avg. P(male) for Group 2 (without Pointy_Nose): 0.9820 ± 0.1009
SIGNIFICANT (p < 1.282e-03): t = 3.83, p = 1.275e-04

--- Attribute: Receding_Hairline ---
Group 1 (Males with Receding_Hairline): 4914 examples
Group 2 (Males without Receding_Hairline): 37136 examples
Avg. P(male) for Group 1 (with Receding_Hairline): 0.9856 ± 0.1010
Avg. P(male) for Group 2 (without Receding_Hairline): 0.9823 ± 0.0984
NOT SIGNIFICANT (p >= 1.282e-03): t = 2.11, p = 3.511e-02

--- Attribute: Rosy_Cheeks ---
Group 1 (Males with Rosy_Cheeks): 132 examples
Group 2 (Males without Rosy_Cheeks): 41918 examples
Avg. P(male) for Group 1 (with Rosy_Cheeks): 0.9788 ± 0.1015
Avg. P(male) for Group 2 (without Rosy_Cheeks): 0.9827 ± 0.0987
NOT SIGNIFICANT (p >= 1.282e-03): t = -0.44, p = 6.622e-01

--- Attribute: Sideburns ---
Group 1 (Males with Sideburns): 5612 examples
Group 2 (Males without Sideburns): 36438 examples
Avg. P(male) for Group 1 (with Sideburns): 0.9981 ± 0.0170
Avg. P(male) for Group 2 (without Sideburns): 0.9803 ± 0.1057
SIGNIFICANT (p < 1.282e-03): t = 29.67, p = 1.783e-191

--- Attribute: Smiling ---
Group 1 (Males with Smiling): 16746 examples
Group 2 (Males without Smiling): 25304 examples
Avg. P(male) for Group 1 (with Smiling): 0.9871 ± 0.0824
Avg. P(male) for Group 2 (without Smiling): 0.9798 ± 0.1081
SIGNIFICANT (p < 1.282e-03): t = 7.83, p = 4.982e-15

--- Attribute: Straight_Hair ---
Group 1 (Males with Straight_Hair): 10134 examples
Group 2 (Males without Straight_Hair): 31916 examples
Avg. P(male) for Group 1 (with Straight_Hair): 0.9862 ± 0.0843
Avg. P(male) for Group 2 (without Straight_Hair): 0.9816 ± 0.1029
SIGNIFICANT (p < 1.282e-03): t = 4.58, p = 4.717e-06

--- Attribute: Wavy_Hair ---
Group 1 (Males with Wavy_Hair): 6018 examples
Group 2 (Males without Wavy_Hair): 36032 examples
Avg. P(male) for Group 1 (with Wavy_Hair): 0.9745 ± 0.1185
Avg. P(male) for Group 2 (without Wavy_Hair): 0.9841 ± 0.0950
SIGNIFICANT (p < 1.282e-03): t = -5.97, p = 2.466e-09

--- Attribute: Wearing_Earrings ---
Group 1 (Males with Wearing_Earrings): 650 examples
Group 2 (Males without Wearing_Earrings): 41400 examples
Avg. P(male) for Group 1 (with Wearing_Earrings): 0.9274 ± 0.2268
Avg. P(male) for Group 2 (without Wearing_Earrings): 0.9836 ± 0.0951
SIGNIFICANT (p < 1.282e-03): t = -6.30, p = 5.508e-10

--- Attribute: Wearing_Hat ---
Group 1 (Males with Wearing_Hat): 3436 examples
Group 2 (Males without Wearing_Hat): 38614 examples
Avg. P(male) for Group 1 (with Wearing_Hat): 0.9753 ± 0.1169
Avg. P(male) for Group 2 (without Wearing_Hat): 0.9834 ± 0.0969
SIGNIFICANT (p < 1.282e-03): t = -3.93, p = 8.587e-05

--- Attribute: Wearing_Lipstick ---
Group 1 (Males with Wearing_Lipstick): 264 examples
Group 2 (Males without Wearing_Lipstick): 41786 examples
Avg. P(male) for Group 1 (with Wearing_Lipstick): 0.8595 ± 0.2582
Avg. P(male) for Group 2 (without Wearing_Lipstick): 0.9835 ± 0.0964
SIGNIFICANT (p < 1.282e-03): t = -7.79, p = 1.591e-13

--- Attribute: Wearing_Necklace ---
Group 1 (Males with Wearing_Necklace): 742 examples
Group 2 (Males without Wearing_Necklace): 41308 examples
Avg. P(male) for Group 1 (with Wearing_Necklace): 0.9466 ± 0.1689
Avg. P(male) for Group 2 (without Wearing_Necklace): 0.9834 ± 0.0969
SIGNIFICANT (p < 1.282e-03): t = -5.91, p = 5.337e-09

--- Attribute: Wearing_Necktie ---
Group 1 (Males with Wearing_Necktie): 7324 examples
Group 2 (Males without Wearing_Necktie): 34726 examples
Avg. P(male) for Group 1 (with Wearing_Necktie): 0.9968 ± 0.0296
Avg. P(male) for Group 2 (without Wearing_Necktie): 0.9797 ± 0.1076
SIGNIFICANT (p < 1.282e-03): t = 25.45, p = 1.053e-141

--- Attribute: Young ---
Group 1 (Males with Young): 26843 examples
Group 2 (Males without Young): 15207 examples
Avg. P(male) for Group 1 (with Young): 0.9803 ± 0.1040
Avg. P(male) for Group 2 (without Young): 0.9870 ± 0.0884
SIGNIFICANT (p < 1.282e-03): t = -6.99, p = 2.807e-12

============================================================
Saved plot: result_imgs_100k/debiasing_comparison.png

============================================================
DEBIASING T-TEST SUMMARY
============================================================
Attribute            | Original t-stat      | Hard Debias t-stat   | Soft Debias t-stat   | |Δ Hard|   | |Δ Soft|
-------------------------------------------------------------------------------------------------------------------
5_o_Clock_Shadow     | 27.36                | -74.08               | 27.26                | 101.43     | 0.10
Arched_Eyebrows      | -1.68                | 9.54                 | -1.65                | 11.23      | 0.04
Attractive           | 4.31                 | -32.30               | 4.32                 | 36.61      | 0.01
Bags_Under_Eyes      | 9.65                 | -5.52                | 9.71                 | 15.17      | 0.06
Bald                 | 17.66                | -32.92               | 17.29                | 50.58      | 0.37
Bangs                | -11.35               | 30.29                | -11.28               | 41.64      | 0.07
Big_Lips             | -4.94                | 17.94                | -4.94                | 22.88      | 0.00
Big_Nose             | 10.75                | 14.19                | 10.83                | 3.44       | 0.08
Black_Hair           | 4.25                 | 14.40                | 4.34                 | 10.14      | 0.09
Blond_Hair           | -6.58                | 1.58                 | -6.64                | 8.15       | 0.06
Blurry               | -8.67                | 1.13                 | -8.74                | 9.80       | 0.07
Brown_Hair           | -0.50                | -5.87                | -0.58                | 5.37       | 0.07
Bushy_Eyebrows       | 13.43                | -11.15               | 13.36                | 24.58      | 0.07
Chubby               | 7.66                 | 18.07                | 7.72                 | 10.41      | 0.06
Double_Chin          | 11.63                | 9.81                 | 11.77                | 1.82       | 0.14
Eyeglasses           | 0.27                 | -0.22                | 0.33                 | 0.49       | 0.07
Goatee               | 29.04                | -5.44                | 29.22                | 34.49      | 0.18
Gray_Hair            | 7.31                 | -28.14               | 7.23                 | 35.46      | 0.08
Heavy_Makeup         | -6.01                | 8.02                 | -6.04                | 14.03      | 0.03
High_Cheekbones      | 4.52                 | 11.61                | 4.60                 | 7.09       | 0.08
Mouth_Slightly_Open  | -0.05                | 8.68                 | 0.11                 | 8.73       | 0.16
Mustache             | 28.01                | 3.49                 | 28.02                | 24.52      | 0.01
Narrow_Eyes          | -2.59                | -1.99                | -2.59                | 0.60       | 0.00
No_Beard             | -27.57               | 42.68                | -27.60               | 70.25      | 0.03
Oval_Face            | 9.05                 | 4.93                 | 9.16                 | 4.12       | 0.11
Pale_Skin            | -5.77                | 4.75                 | -5.83                | 10.51      | 0.07
Pointy_Nose          | 3.96                 | -24.75               | 3.83                 | 28.71      | 0.12
Receding_Hairline    | 2.16                 | -0.71                | 2.11                 | 2.87       | 0.05
Rosy_Cheeks          | -0.49                | -3.93                | -0.44                | 3.45       | 0.05
Sideburns            | 29.76                | -33.72               | 29.67                | 63.48      | 0.09
Smiling              | 7.72                 | 1.14                 | 7.83                 | 6.58       | 0.11
Straight_Hair        | 4.49                 | -0.32                | 4.58                 | 4.81       | 0.09
Wavy_Hair            | -5.94                | 3.94                 | -5.97                | 9.87       | 0.03
Wearing_Earrings     | -6.26                | 10.67                | -6.30                | 16.93      | 0.03
Wearing_Hat          | -3.90                | 9.67                 | -3.93                | 13.57      | 0.03
Wearing_Lipstick     | -7.83                | 11.59                | -7.79                | 19.42      | 0.05
Wearing_Necklace     | -5.88                | 7.82                 | -5.91                | 13.70      | 0.02
Wearing_Necktie      | 25.25                | -1.09                | 25.45                | 26.35      | 0.19
Young                | -6.98                | -17.22               | -6.99                | 10.24      | 0.01

============================================================
Analysis complete! All results saved to result_imgs_100k/

Generated plots:
  - attribute_bias_all_with_arrows.png
  - attribute_bias_directions_original.png
  - attribute_bias_selected_only.png
  - debiasing_comparison.png
  - gender_pca_custom.png
  - gender_tsne_custom.png
  - male_groups_confidence_hist_Attractive.png
  - male_groups_confidence_hist_Gray_Hair.png
  - male_groups_confidence_hist_No_Beard.png
  - male_groups_confidence_hist_Wearing_Lipstick.png
  - male_groups_confidence_hist_Young.png
============================================================
