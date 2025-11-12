import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from loguru import logger

import joblib
import tqdm

from datacat4ml.Scripts.const import PREP_DATA_DIR

"""
Extract assay features like LSA. 
A list of GPCR assay ids is provided, and a Numpy dense matrix is created such that its i-th row contains the features of the i-th assay in the initial list.

For FS-Mol, we use the following columns:
`python encode_assay.py --assay_parquet_path=assay_info.parquet --encoding=clip --gpu=0 --columns \
assay_type_description description assay_category assay_cell_type assay_chembl_id assay_classification assay_organism assay_parameters assay_strain assay_subcellular_fraction assay_tax_id assay_test_type assay_tissue assay_type bao_format bao_label cell_chembl_id confidence_description confidence_score document_chembl_id relationship_description relationship_type src_assay_id src_id target_chembl_id tissue_chembl_id variant_sequence \
--suffix=all`
"""

#======================== clip_encode ========================
def clip_encode(list_of_assay_descriptions, gpu=0, batch_size=2048, truncate=True, verbose=True):
    """
    Encode a list of assay descriptions using a fitted Model.
    It is supposed to be called once.

    Params
    ------
    list_of_assay_descriptions: list of strings
        List of assay descriptions to be encoded.
    gpu: int
        Device to use for the CLIP model.
    batch_size: int
        Batch size to use for the CLIP model.
    truncate: bool
        default: True
        Whether to truncate the assay descriptions to 77 tokens (truncated from the end, because the beginning of a sentence often contains the most relevant information), the default setting for CLIP.
    
    Returns
    -------
    numpy.ndarray
        Numpy dense matrix with shape (n_assays, n_components). # n_components is the size of vector representation for each assay description.
    """
    import torch
    import clip
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Load CLIP model on {device}.')
    model, preprocess = clip.load("ViT-B/32", device=device) # adopted from clip repo

    logger.info('Encode assay descriptions using CLIP.')
    with torch.no_grad(): 
        text_features = []
        for b in tqdm.tqdm(range(0, len(list_of_assay_descriptions), batch_size), desc='Encode assay descriptions', disable=not verbose):
            tokenized_text = clip.tokenize(list_of_assay_descriptions[b:min(b+batch_size, len(list_of_assay_descriptions))], truncate=truncate).to(device)
            tf = model.encode_text(tokenized_text)
            text_features.append(tf.cpu().detach().numpy()) # `.cpu()`: move the tensor from GPU to CPU; `.detach()`: no longer track the gradient; `.numpy()`: convert the tensor to a numpy array
    text_features = np.concatenate(text_features, axis=0)
    
    return text_features.astype(np.float32)

#======================== lsa_encode ========================
def lsa_fit(list_of_assay_descriptions, model_save_path='./data/models/lsa.joblib', n_components=355, verbose=True): #Yu? change the value of model_save_path later
    """
    Fit a sklearn TruncatedSVD model using a list of assay descriptions.

    Params
    ------
    list_of_assay_descriptions: list of strings
        List of assay descriptions to be encoded.
    model_save_path: str
        Path to save the fitted sklearn LSA model in joblib format.
    n_components: int
        Number of components to use for the TruncatedSVD model.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    logger.info('Set up and fit-transform a sklearn TfidfVectorizer.')
    #tok = Tokenizer()
    
    tfidf = TfidfVectorizer(
        strip_accents='unicode',
        analyzer='word',
        #tokenizer=tok, #Yu?: enable it will lead to error:'LookupError: Resource punkt_tab not found.' Compare the tok and default tokenizer of TfidfVectorizer. Otherwise, just use the default tokenizer.
        stop_words='english',
        max_df=0.95,
        min_df =1 / 10000,
        dtype =np.float32
    )

    features = tfidf.fit_transform(list_of_assay_descriptions)
    logger.info(f'tfidf.vocabulary size: {len(tfidf.vocabulary_)}')
    #if verbose:
    #    logger.info('Fit a sklearn TruncatedSVD model with {n_components} components.'.format(n_components))
    svd = TruncatedSVD(n_components=n_components)
    print(f'SVD model initialized with {n_components} components.')
    svd.fit(features)

    model = Pipeline([('tfidf', tfidf), ('svd', svd)])
    if verbose:
        logger.info('Save the fitted model.')
    # check if model_save_path exists otherwise create it
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True) #Yu?
    joblib.dump(model, model_save_path)

    return model

def lsa_encode(list_of_assay_descriptions, lsa_path='', verbose=True):
    """
    Encode a list of assay descriptions using a fitted LSA model.

    Params
    ------
    list_of_assay_descriptions: list of strings
        List of assay descriptions to be encoded.
    lsa_path: str
        Path to a fitted sklearn LSA model in joblib format.
    n_components: int
        Number of components to use for the TruncatedSVD model.

    Returns
    -------
    numpy.ndarray
        Numpy dense matrix with shape (n_assays, n_components).
    """
    if verbose:
        logger.info('Load a fitted LSA model.')
    model = joblib.load(lsa_path)

    if verbose:
        logger.info('Encode assay descriptions using LSA.')
    features = model.transform(list_of_assay_descriptions)

    return features


# ========================= Main function =========================
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Computer features for a collection of GPCR assay descriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--assay_parquet_path', default='assay_info.parquet', help='Path to a parquet file with assay index to AID for which to extract features.')
    #parser.add_argument('-c', '--columns', nargs='+', help='Columns to use for the assay description. default: title and subtitle', default=['title', 'subtitle'])
    parser.add_argument('--columns_list', help="List of columns to use for the assay info. default: 'columns_short'", default='columns_short')
    parser.add_argument('--suffix', help='Suffix to add to the output file.', default=None)
    parser.add_argument('--encoding', help='Encoding-type to use for the assay descriptions. Available are text, clamp, lsa, default:lsa', default='lsa') 
    parser.add_argument('--lsa_path', help='Path to a fitted sklearn TfidfVectorizer+LSA model in joblib format, or where to save it if not present.', default='./data/models/lsa.joblib') #Yu? change the value for default
    parser.add_argument('--train_set_size', help='The ratio of assay descriptions for training the model to the whole dataset. range: 0-1, default: first 80%', default=0.8, type=float)
    parser.add_argument('--gpu', help='GPU number to use for a GPU-based encoding, if any. Default:0', default=0)
    parser.add_argument('--batch_size', help='Batch size to use for a GPU-based encoding. default: 2048', default=2048, type=int) #Yu? why default 2048
    parser.add_argument('--n_components', help='Number of components to use for the TruncatedSVD model. default:355', default=355, type=int) #Yu? why default 355
    args = parser.parse_args()

    df = pd.read_parquet(args.assay_parquet_path)
    # make a folder 'encoded_assay' in `args.assay_parquet_path` if not exists
    path = os.path.join(os.path.dirname(args.assay_parquet_path), 'encoded_assays')
    os.makedirs(path, exist_ok=True)

    print(f'================= columns_list: {args.columns_list} =================')

    if args.columns_list == 'assay_desc_only':
        columns = ['assay_description']
    elif args.columns_list=='columns_short':
        columns = ["assay_idx", "assay_chembl_id", "assay_description"]
    elif args.columns_list=='columns_middle':
        columns = ["assay_idx", "assay_id", "assay_chembl_id", "assay_description", "assay_type", "assay_type_description", "assay_category", "assay_organism", "assay_tax_id", "assay_strain", "assay_tissue", "assay_cell_type", "assay_subcellular_fraction", "bao_format", "bao_label", "variant_id", "assay_test_type", "cell_id", "tissue_id", "relationship_type_description", "aidx", "confidence_score_description", "tid"]
    elif args.columns_list=='columns_long':
        columns = ["assay_idx", "assay_id", "assay_chembl_id", "assay_description", "assay_type", "assay_type_description", "assay_category", "assay_organism", "assay_tax_id", "assay_strain", "assay_tissue", "assay_cell_type", "assay_subcellular_fraction", "bao_format", "bao_label", "variant_id", "assay_test_type", "cell_id", "tissue_id", "relationship_type_description", "aidx", "confidence_score_description", "tid", "target_chembl_id", "effect_description", "assay_keywords_description"]
    elif args.columns_list=='columns_full':
        columns = ["assay_idx", "assay_id", "assay_chembl_id", "assay_description", "assay_type", "assay_type_description", "assay_category", "assay_organism", "assay_tax_id", "assay_strain", "assay_tissue", "assay_cell_type", "assay_subcellular_fraction", "bao_format", "bao_label", "variant_id", "assay_test_type", "cell_id", "tissue_id", "relationship_type", "relationship_type_description", "aidx", "confidence_score", "confidence_score_description", "tid", "target_chembl_id", "effect", "effect_description", "assay", "assay_keywords_description"]

    # check if all columns are present
    if not all([c in df.columns for c in columns]):
        raise ValueError(f'Columns {columns} not found in the assay dataframe. Available columns: {df.columns}')
    
    df[columns] = df[columns].fillna('')  # fill NaN with empty string
    df[columns] = df[columns].astype(str)  # convert all columns to string

    list_of_assay_descriptions = df[columns].apply(
        lambda x: ' '.join([f"{col}:{val}" for col, val in x.items()]), axis=1).tolist()

    logger.info(f'example assay description: {list_of_assay_descriptions[0]}')

    print(f'================== encoding: {args.encoding} =================')

    if args.encoding == 'text':
        features = np.array(list_of_assay_descriptions)
    elif args.encoding == 'lsa':
        logger.info('Encode assay descriptions using LSA')
        # load model if the file exists
        if not Path(args.lsa_path).is_file():
            logger.info('Fit a sklearn TfidfVectorizer model on training data.')
            # lsa_save_path = path.with_name(f'assay_lsa_enc{"_"+args.suffix if args.suffix else ""}.joblib')
            logger.info(f'Save the fitted LSA-model to {args.lsa_path}, load it later using the argument --lsa_path')

            # Todo custom fit depending on training-set size
            train_set_size = args.train_set_size
            train_set_size = int(len(list_of_assay_descriptions)*train_set_size)
            logger.info(f'Fit on {train_set_size} train assay descriptions, {train_set_size/len(list_of_assay_descriptions)*100:.2f}% of the data.')
            model_save_path = os.path.join(args.lsa_path, f'{args.encoding}_{args.columns_list}_lsa.joblib')
            print(f'model_save_path: {model_save_path}')
            print(f'n_components: {args.n_components}')
            model = lsa_fit(list_of_assay_descriptions[:int(train_set_size)], model_save_path=model_save_path, n_components=args.n_components)
        
        features = lsa_encode(list_of_assay_descriptions, lsa_path=model_save_path)

    elif args.encoding == 'clip':
        features = clip_encode(list_of_assay_descriptions, gpu=args.gpu, verbose=True, batch_size=args.batch_size)
    elif args.encoding == 'biobert': #Yu? remove it later
        raise NotImplementedError('Biobert encoding not implemented yet.') 
    else:
        raise ValueError(f'Encoding {args.encoding} not implemented')
    
    print(f'Path(path) is' + str(Path(path)))
    fn= os.path.join(path, f'assay_features_{args.encoding}_{args.columns_list}{"_"+args.suffix if args.suffix else ""}.npy')
    np.save(fn, features)
    logger.info(f'Saved assay features to {fn}')