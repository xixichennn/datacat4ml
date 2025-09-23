import os
from typing import List, Dict
import numpy as np
import pandas as pd
import argparse


from datacat4ml.const import FEAT_DATA_DIR, FEAT_HHD_OR_DIR , FEAT_MHD_OR_DIR , FEAT_LHD_OR_DIR, FEAT_HHD_GPCR_DIR, FEAT_MHD_GPCR_DIR, FEAT_LHD_GPCR_DIR
from datacat4ml.const import CAT_FIG_DIR, CURA_FIG_DIR, FEAT_FIG_DIR
from datacat4ml.utils import mkdirs

# ===================== plotting =====================
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#======================= t-SNE =======================
from openTSNE import TSNE
from adjustText import adjust_text


#======================== palette ========================
# activity
one_color = '#8B006B' # i.e. active; dark red
zero_color = '#538CBA' # i.e. inactive; teal
activity_palette = {1:one_color, 0:zero_color}

# target_chembl_id
mor_color = '#8B006B'
kor_color = '#538CBA'
dor_color = '#A5C93D'
nor_color = '#2000D7'
target_palette = {
    'CHEMBL233':mor_color,
    'CHEMBL237':kor_color,
    'CHEMBL236':dor_color,
    'CHEMBL2014':nor_color,
    }

# 'effect'
agon_color = '#f5426c'
antag_color = '#4278f5'
bind_color = '#b0e3e6'
effect_palette = {'agon':agon_color, 'antag':antag_color, 'bind':bind_color}

# assay_chembl_id
def set_assay_chembl_id_palette(assay_chembl_ids: List[str]) -> Dict[str, str]:
    n = len(assay_chembl_ids)
    palette = sns.color_palette("Spectral", n)
    assay_chembl_id_palette = {assay_chembl_id:palette[i] for i, assay_chembl_id in enumerate(assay_chembl_ids)}
    return assay_chembl_id_palette

##########################
y_palette = {
    "target_chembl_id": target_palette,
    "activity": activity_palette,
    "effect": effect_palette,
}
#=======================================================

def concat_pkl(in_path=FEAT_MHD_OR_DIR) -> pd.DataFrame:
    concat_df = pd.DataFrame()
    for f in os.listdir(os.path.join(in_path, 'all')):
        df = pd.read_pickle(os.path.join(in_path, 'all', f))
        concat_df = pd.concat([concat_df, df], ignore_index=True)
    return concat_df

def plot(
    x, # x is the fitted 2D t-SNE coordinates
    y, # y is the categorical labels for coloring
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    savepath=None,
    figname=None,
    **kwargs
):

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    if title is not None:
        ax.set_title(title, fontsize=16)

    plot_params = {"alpha": kwargs.get("alpha", 1), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)

    # Assign colors
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
    point_colors = list(map(colors.get, y))
    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            texts = []
            for idx, label in enumerate(classes):
                texts.append(
                    ax.text(
                        centers[idx, 0],
                        centers[idx, 1] + 2.2,
                        label,
                        fontsize=kwargs.get("fontsize", 8),
                        horizontalalignment="center",
                    )
                )
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))

    # Hide ticks and axis
    #ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    # # Instead, add labels
    #ax.set_xlabel("t-SNE 1")
    #x.set_ylabel("t-SNE 2")

    # tick marks but no box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                #markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, fontsize=12, **legend_kwargs_)
    
    # Save figure if path is given
    if savepath is not None:
        fig = ax.get_figure()
        fig.savefig(os.path.join(savepath, figname), bbox_inches="tight")

def run_tsne_plot(df, x_col="ECFP4", y_col="activity",
                  perplexity=30, 
                  metric="euclidean", # other options: "cosine"
                  initialization="pca",  # other option: "random"
                  exaggeration=4, # This can be used to form more densely packed clusters and is useful for large data sets.
                  colors=activity_palette,
                  draw_legend=True,
                  draw_centers=False,
                  draw_cluster_labels=False,
                  savepath=None,
                  figname=None,
                ):
    # check if there is array of 0s in x_col
    if (df[x_col].apply(lambda x: np.all(np.array(x) == 0)).any()):
        print(f"Warning: There are some rows with all-zero {x_col} in the dataframe. Please check and remove them before t-SNE.")
        df = df[~df[x_col].apply(lambda x: np.all(np.array(x) == 0))]
        print(f"After removing all-zero {x_col}, the dataframe has {len(df)} rows.")

    X = np.vstack(df[x_col].values)   # stack fingerprints into a 2D numpy array
    y = df[y_col].values    # categorical labels for coloring

    # Run openTSNE
    tsne = TSNE(n_components=2,random_state=42, n_jobs=4,# use multiple cores, adjust as needed
                perplexity=perplexity, 
                metric=metric,
                initialization=initialization,  # PCA initialization can help with reproducibility
                exaggeration=exaggeration, # use larger exaggeration for larger datasets
    )

    embedding = tsne.fit(X)
    if savepath is not None:
        if figname is None: # for concat_df
            figname = f"tsne_{(savepath.split('/')[-1]).replace('feat_', '')}_{x_col}_{y_col}.pdf"
        plot(x=embedding, y=y, title=f'{x_col}', colors=colors, 
             draw_legend=draw_legend, draw_centers=draw_centers, draw_cluster_labels=draw_cluster_labels,
            savepath=savepath, figname=figname)
    elif savepath is None:
        plot(x=embedding, y=y, title=f'{x_col}', colors=colors, 
             draw_legend=draw_legend, draw_centers=draw_centers, draw_cluster_labels=draw_cluster_labels)

def main(descriptor: str = "ECFP4"):

    print(f'Processing descriptor: {descriptor} ...')

    print(f'For concat_df ......\n')
    #===========================================================
    # y_col='assay_chembl_id', only for in_path: FEAT_LHD_OR_DIR
    #===========================================================
    lhd_or_df = concat_pkl(FEAT_LHD_OR_DIR)
    save_path = os.path.join(FEAT_FIG_DIR, 'feat_lhd_or')
    mkdirs(save_path)

    # Get counts and filter valid IDs
    id_counts = lhd_or_df['assay_chembl_id'].value_counts()
    assay_chembl_ids = id_counts[id_counts >= 50].index.tolist()
    palette = set_assay_chembl_id_palette(assay_chembl_ids)

    new_lhd_or_df = lhd_or_df[lhd_or_df['assay_chembl_id'].isin(assay_chembl_ids)]

    run_tsne_plot(new_lhd_or_df, x_col=descriptor, y_col='assay_chembl_id',
                perplexity=30,
                metric="euclidean", # options: "cosine", "euclidean"
                initialization="pca",  # options: "random", "pca"
                exaggeration=1, # use larger exaggeration for larger datasets
                colors=palette,
                draw_legend=False, draw_centers=True, draw_cluster_labels=True,
                savepath=save_path
                )

    #===================================================
    #  y_col= ..., in_path: FEAT_MHD_OR_DIR
    #===================================================
    for y_col in ["target_chembl_id",
                "activity",
                "effect",
                ]:
        print(f'Processing label: {y_col} ...')

        # input datasets
        mhd_or_df = concat_pkl(FEAT_MHD_OR_DIR)
        save_path = os.path.join(FEAT_FIG_DIR, 'feat_mhd_or')
        mkdirs(save_path)

        run_tsne_plot(mhd_or_df, x_col=descriptor, y_col=y_col,
                    perplexity=30, 
                    metric="euclidean", # options: "cosine", "euclidean"
                    initialization="pca",  # options: "random", "pca"
                    exaggeration=1, # use larger exaggeration for larger datasets
                    colors=y_palette[y_col],
                    draw_legend=False,
                    savepath=save_path
                    )
        
    #=======================================================================================
    # y_col='activity' for each single pkl in in_path: FEAT_LHD_OR_DIR and FEAT_MHD_OR_DIR
    #=======================================================================================
    print(f'\nFor each single pkl ......\n')
    in_paths = [FEAT_MHD_OR_DIR, FEAT_LHD_OR_DIR]
    for in_path in in_paths:
        print(f'Processing in_path: {in_path} ...')
        for f in os.listdir(os.path.join(in_path, 'all')):
            df = pd.read_pickle(os.path.join(in_path, 'all', f))
            save_path = os.path.join(FEAT_FIG_DIR, in_path.split('/')[-1])

        run_tsne_plot(df, x_col=descriptor, y_col='activity',
                perplexity=30,
                metric="euclidean", # options: "cosine", "euclidean"
                initialization="pca",  # options: "random", "pca"
                exaggeration=1, # use larger exaggeration for larger datasets
                colors=activity_palette,
                draw_legend=False,
                savepath=save_path, figname=f"tsne_{f.replace('_featurized.pkl', '')}_{descriptor}_activity.pdf"
                )
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE and plot")
    parser.add_argument("--descriptor", required=True, help="descriptor column name, e.g. ECFP4")
    
    args = parser.parse_args()

    main(args.descriptor)