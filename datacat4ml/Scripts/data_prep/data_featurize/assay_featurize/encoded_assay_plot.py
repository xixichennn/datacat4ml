# conda env: datacat4ml (python 3.8)
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from datacat4ml.Scripts.const import SPLIT_DATA_DIR

# load the .npy file
npy_path_folder = os.path.join(SPLIT_DATA_DIR, 'fsmol_alike', 'MHDsFold', 'encoded_assays')

text_short = np.load(os.path.join(npy_path_folder, 'assay_features_text_columns_short.npy'))
text_middle = np.load(os.path.join(npy_path_folder, 'assay_features_text_columns_middle.npy'))
text_long = np.load(os.path.join(npy_path_folder, 'assay_features_text_columns_long.npy'))
text_full = np.load(os.path.join(npy_path_folder, 'assay_features_text_columns_full.npy'))

clip_short = np.load(os.path.join(npy_path_folder, 'assay_features_clip_columns_short.npy'))
clip_middle = np.load(os.path.join(npy_path_folder, 'assay_features_clip_columns_middle.npy'))
clip_long = np.load(os.path.join(npy_path_folder, 'assay_features_clip_columns_long.npy'))
clip_full = np.load(os.path.join(npy_path_folder, 'assay_features_clip_columns_full.npy'))

lsa_short = np.load(os.path.join(npy_path_folder, 'assay_features_lsa_columns_short.npy'))
lsa_middle = np.load(os.path.join(npy_path_folder, 'assay_features_lsa_columns_middle.npy'))
lsa_long = np.load(os.path.join(npy_path_folder, 'assay_features_lsa_columns_long.npy'))
lsa_full = np.load(os.path.join(npy_path_folder, 'assay_features_lsa_columns_full.npy'))

# ================= text features =================
def text_features_plot(text_feature=text_short):
    """Plot the distribution of text lengths in the text features"""
    

    if text_feature is text_short:
        text_feature_name = "text_short"
    elif text_feature is text_middle:
        text_feature_name = "text_middle"
    elif text_feature is text_long:
        text_feature_name = "text_long"
    elif text_feature is text_full:
        text_feature_name = "text_full"
    

    text_lengths = np.array([len(s) for s in text_feature])

    # plot hisgram
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Text Lengths in {text_feature_name}")
    plt.xlabel("Length of Text")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(SPLIT_DATA_DIR, 'fsmol_alike', 'MHDsFold', 'encoded_assays', f"{text_feature_name}_length_histogram.png"))

# ================= clip or lsa features =================
def clip_lsa_features_plot(feature=clip_short):
    """Using PCA or t-SNE to visualize the features"""
    if feature is clip_short:
        feature_name = "clip_short"
    elif feature is clip_middle:
        feature_name = "clip_middle"
    elif feature is clip_long:
        feature_name = "clip_long"
    elif feature is clip_full:
        feature_name = "clip_full"
    elif feature is lsa_short:
        feature_name = "lsa_short"
    elif feature is lsa_middle:
        feature_name = "lsa_middle"
    elif feature is lsa_long:
        feature_name = "lsa_long"
    elif feature is lsa_full:
        feature_name = "lsa_full"

    # Apply PCA to 2D for visualization
    pca = PCA(n_components=2).fit_transform(feature)

    # Plot CLIP embeddings
    plt.figure(figsize=(10, 6))
    plt.scatter(pca[:, 0], pca[:, 1], s=1, alpha=0.5)
    plt.title(f"{feature_name} Embeddings (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(SPLIT_DATA_DIR, 'fsmol_alike', 'MHDsFold', 'encoded_assays',f"{feature_name}_pca.png"))

# ================= Main function =================
if __name__ == "__main__":
    # Plot text features
    text_features_plot(text_short)
    text_features_plot(text_middle)
    text_features_plot(text_long)
    text_features_plot(text_full)

    # Plot CLIP features
    clip_lsa_features_plot(clip_short)
    clip_lsa_features_plot(clip_middle)
    clip_lsa_features_plot(clip_long)
    clip_lsa_features_plot(clip_full)

    # Plot LSA features
    clip_lsa_features_plot(lsa_short)
    clip_lsa_features_plot(lsa_middle)
    clip_lsa_features_plot(lsa_long)
    clip_lsa_features_plot(lsa_full)