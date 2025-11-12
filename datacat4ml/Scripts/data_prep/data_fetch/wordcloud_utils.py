# conda environment: base(Python 3.12.4)
# Error 'Only supported for TrueType fonts' occurs when using pyg env
import sys
sys.path.append('/storage/homefs/yc24j783/datacat4ml/')
from datacat4ml.Scripts.const import DATA_DIR, FIG_DIR, FETCH_DATA_DIR , FETCH_FIG_DIR, OR_chembl_ids
from datacat4ml.Scripts.utils import mkdirs

import os
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def gen_wordcloud(text, use_mask=False, additional_stopwords=None, output_file='word_cloud.png'):
    """ 
    Generate a word cloud from a given text.

    Parameters
    ----------
    text : str
        Text to generate word cloud from.
    use_mask : bool
        Whether to use the gpcr mask image for the word cloud.
    additional_stopwords : list of str
        Additional stopwords to remove from text.
    output_file : str
        Output file name for the word cloud image.

    """
    # Find an image to make with
    gpcr_image = Image.open(os.path.join(DATA_DIR, 'GPCR_struc.png'))
    # Create mask with image path
    gpcr_mask = np.array(gpcr_image)

    # Add additional stopwords
    stopwords = set(STOPWORDS)
    if additional_stopwords:
        for word in additional_stopwords:
            stopwords.add(word)
    
    # Create and generate word cloud
    if use_mask:
        wordcloud = WordCloud(background_color="white", mask=gpcr_mask, collocations=False, stopwords=stopwords, contour_color="white", contour_width=1)
    else:
        wordcloud = WordCloud(background_color="white", collocations=False, stopwords=stopwords, contour_color="white", contour_width=1)
    wordcloud.generate(text)

    image_colors = ImageColorGenerator(gpcr_mask)

    # Show word cloud on pyplot figure
    plt.imshow(wordcloud, interpolation="bilinear")
    #plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()

    # save the word cloud image as a svg file
    filepath = os.path.join(FETCH_FIG_DIR, 'wordcloud')
    mkdirs(filepath)
    wordcloud.to_file(os.path.join(filepath, output_file))
