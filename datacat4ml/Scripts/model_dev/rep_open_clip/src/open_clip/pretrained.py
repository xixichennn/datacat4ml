# conda env: pyg (Python 3.9.16)
import copy
import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Iterable, Optional, Union

from tqdm import tqdm


try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

from datacat4ml.Scripts.model_dev.rep_open_clip.src.open_clip.version import __version__
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False

#################   Yu's ##########################

from datacat4ml.Scripts.model_dev.rep_open_clip.src.open_clip.constants import (
#-->     IMAGENET_MEAN,
#-->     IMAGENET_STD,
#-->     INCEPTION_MEAN,
#-->     INCEPTION_STD,
#-->     OPENAI_DATASET_MEAN,
#-->     OPENAI_DATASET_STD,
    HF_WEIGHTS_NAME,
    HF_SAFE_WEIGHTS_NAME,
)


#################   Yu's ##########################


################### `download_pretrained_from_hf` ##############################
def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub

def _get_safe_alternatives(filename: str) -> Iterable[str]:
    """Returns potential safetensors alternatives for a given filename.

    Use case:
        When downloading a model from the Huggingface Hub, we first look if a .safetensors file exists and if yes, we use it.
    """
    if filename == HF_WEIGHTS_NAME:
        yield HF_SAFE_WEIGHTS_NAME

    if filename not in (HF_WEIGHTS_NAME,) and (filename.endswith(".bin") or filename.endswith(".pth")):
        yield filename[:-4] + ".safetensors"

def download_pretrained_from_hf(
        model_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
):
    has_hf_hub(True)

    filename = filename or HF_WEIGHTS_NAME

    # Look for .safetensors alternatives and load from it if it exists
    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                cached_file = hf_hub_download(
                    repo_id=model_id,
                    filename=safe_filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                return cached_file
            except Exception:
                pass

    try:
        # Attempt to download the file
        cached_file = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        return cached_file  # Return the path to the downloaded file if successful
    except Exception as e:
        raise FileNotFoundError(f"Failed to download file ({filename}) for {model_id}. Last error: {e}")
