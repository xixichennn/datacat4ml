# conda env: datacat (python=3.8.2)
from pathlib import Path
from loguru import logger

import torch
from torch.utils.data import Dataset
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse


try: # only if Graph-Model is used
    import dgl
except: pass

#================================================================================================================ 
# ML dataloader
# =============================================================================================================== 


#================================================================================================================ 
# CL dataloader
# =============================================================================================================== 
def get_sparse_indices_and_data(m, i):
    """Get the indices and data of a sparse matrix.
    This function will be called in `InMemoryClamp.getitem_meta_assay`
    
    Params:
    -------
    m: a sparse matrix.
    i: the index of the row for which to extract the non-zero elements.

    Returns:
    -------
    tuple: (indices, data)
        col_indices: the column indices of the non-zero data in row i. (in csr format, only the non-zero elements are stored)
        data: the values of the non-zero elements in row i.
    """
    # `m.data`: the non-zero values of the sparse matrix
    # `m.indices`: the column indices of the non-zero values
    # `m.indptr`: which maps the elements of `data` and `indices` to the rows of the sparse matrix. Explanation: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    col_indices = m.indices[m.indptr[i]:m.indptr[i+1]] 
    data = m.data[m.indptr[i]:m.indptr[i+1]]
    return col_indices, data

class InMemoryClamp(Dataset):
    """
    Subclass of :class:`torch.utils.data.Dataset` holding activity data, 
    that is, activity triplets, and compound and assay feature vectors. 
    Notice that the compound and assay features are already precompouted and stored in the `encoded_compounds` and `encoded_assays` directories, respectively.
    The current implementation of this class doesn't support feature computation on the fly.

    :class:`InMemoryClamp` supports two different indexing (and iteration) styles. 
    The default style is to itreate over `(compound, assay, activity)` COO triplets, however they are sorted.
    The "meta-assays" style consists in interating over unique compounds using a CSR sparse structure,
    and averaging the feature vectors of the positive and negative assays of each compound. 

    By inheriting from :class:`torch.utils.data.Dataset`, this class must implement at least two methods:
    - :meth:`__len__` to return the size of the dataset.
    - :meth:`__getitem__` to retrieve a single data point from the dataset.
    """

    def __init__(
            self,
            root: Union[str, Path],
            assay_mode: str,
            assay_column_list: str = 'columns_short', 
            compound_mode: str = None, 
            train_size: float = 0.6, # -->self._chunk -->def _find_splits; The train_valid_test ratio would be 6:2:2  ?Yu: it's used yet in current workflow.
            verbose: bool = True
    ) -> None:
        """
        Instantiate the dataset class.

        - The data is loaded in memory with the :meth:`_load_dataset` method.
        - Splits are created separately along compounds and along assays with the :meth:`_find_splits` method. 
            Compound and assay splits can be interwoven with the :meth:`subset` method. 

        Params:
        root: str or :class:`pathlib.Path`
            Path to a directory of ready activity files.
        assay_mode: str
            Type of assay features ("clip", or "lsa").
        assay_column_list: str
            The column list to use for assay features. Default is 'columns_short'.
        train_size: float (between 0 and 1)
            Fraction of compounds and assays assigned to training data.
        verbose: bool
            Be verbose if True.
        """
        self.root = Path(root)
        self.assay_mode = assay_mode
        self.assay_column_list = assay_column_list
        self.compound_mode = compound_mode
        self.train_size = train_size
        self.verbose = verbose

        self._load_dataset()
        self._find_splits() 

        self.meta_assays = False #Yu: remove later if not used.

    def _load_dataset(self) -> None:
        """
        Load prepared dataset from the `root` directory:

        - `activity`: Parquet file containing `(compound, assay, activity)` triplets. Compounds and assays are represented by indices, 
        and thus the file is directly loaded into a :class:`scipy.sparse.coo_matrix` with rows corresponding to compounds and columns corresponding to assays.

        - `compound_names`: Parquet file containing the mapping between the compound index used in `activity` and the corresponding compound name.
        It is loaded into a :class:`pandas.DataFrame`.

        - `assay_names`: Parquet file containing the mapping between the assay index used in `activity` and the corresponding assay name. 
        It is loaded into a :class:`pandas.DataFrame`.

        - `compound_features`: npz file containing the compound features array, where the feature vector for the compound indexed by `idx` is stored in the `idx`-th row. 
        It is loaded into a :class:`scipy.sparse.csr_matrix`.

        - `assay_features`: npy file containing the assay features array, where the feature vector for the assay indexed by `idx` is stored in the `idx`-th row.
        It is loaded into a :class:`numpy.ndarray`.

        Compute the additional basic dataset attributes `num_compounds`, `num_assays`, `compound_features_size`, `assay_features_size`.
        """

        if self.verbose:
            logger.info(f'Load dataset from "{self.root} with {self.assay_mode}" assay features.')

        #======= Load compound data =======
        with open(self.root / 'compound_names.parquet', 'rb') as f:
            self.compound_names = pd.read_parquet(f)
        self.num_compounds = len(self.compound_names)

        #compound_modes = self.compound_mode.split('+') if self.compound_mode is not None else 1
        #if len(compound_modes) > 1:
        #    logger.info('Multiple compound modes are concatenated ')
        #    self.compound_features = np.concatenate([self._load_compound(cm) for cm in compound_modes], axis=1)
        #else:
        self.compound_features = self._load_compound(self.compound_mode)
        # compound_features_size
        if 'graph' in self.compound_mode and (not 'graphormer' in self.compound_mode):
            self.compound_features_size = self.compound_features[0].ndata['h'].shape[1] # in_edge_feats. #? Yu: could be removed if graph is not used finally.
        elif isinstance(self.compound_features, pd.DataFrame):
            self.compound_features_size = 40000 #?Yu: too large?
        else:
            if len(self.compound_features.shape)>1:
                self.compound_features_size = self.compound_features.shape[1]
            else: #In this case, self.compound_features is a 1D array, then it's treated as having only one features per compound.
                self.compound_features_size = 1 

        #======== Load assay data ========
        with open(self.root / 'assay_info.parquet', 'rb') as f:
            self.assay_names = pd.read_parquet(f)
        self.num_assays = len(self.assay_names)

        assay_modes = self.assay_mode.split('+')
        if len(assay_modes)>1:
            logger.info('Multiple assay modes are concatenated')
            self.assay_features = np.concatenate([self._load_assay(am, self.assay_column_list) for am in assay_modes], axis=1)
        else:
            self.assay_features = self._load_assay(self.assay_mode, self.assay_column_list)

        # assay_features_size
        if (self.assay_features is None):
            self.assay_features_size = 512 #wild guess also 512
        elif len(self.assay_features.shape)==1:
            # its only a list, so usually text
            self.assay_features_size = 768  # a common default for models like BERT.
        else: # usually a 2D array
            self.assay_features_size = self.assay_features.shape[1]

        #======= Load activity data =======
        with open(self.root / 'activity.parquet', 'rb') as f:
            activity_df = pd.read_parquet(f)
            self.activity_df = activity_df
        
        self.activity = sparse.coo_matrix(
            (
                activity_df['activity'],# activity is the value
                (activity_df['compound_idx'], activity_df['assay_idx']) # compound in row, assay in column.
            ),
            shape=(self.num_compounds, self.num_assays),
        )
    
    def _load_compound(self, compound_mode=None):
        cmpfn = f'encoded_compounds/compound_features{"_"+compound_mode if compound_mode else ""}'
        logger.info(f'cmpfn: {cmpfn}')
        #?Yu: if 'graph' is not used, remove the below code
        if 'graph' in compound_mode and (not 'graphormer' in compound_mode):
            logger.info(f'graph in compound mode: loading '+cmpfn)
            import dgl
            from dgl.data.utils import load_graphs
            compound_features = load_graphs(str(self.root/(cmpfn+".bin")))[0]
            compound_features = np.array(compound_features)
        elif compound_mode == 'smiles':
            compound_features = pd.read_parquet(self.root/('compound_smiles.parquet'))['CanonicalSMILES'].values
        else:
            try: #tries to open npz files else npy
                with open(self.root/(cmpfn+".npz"), 'rb') as f:
                    compound_features = sparse.load_npz(f)
            except:
                logger.info(f'loading '+cmpfn+'.npz failed, using .npy instead')
                try:
                    logger.info(f'loading '+cmpfn+'.npy')
                    compound_features = np.load(self.root/(cmpfn+".npy"))
                except:
                    logger.info(f'loading '+cmpfn+'.npy failed, trying to compute it on the fly')
                    compound_features = pd.read_parquet(self.root/('compound_smiles.parquet'))
        return compound_features

    def _load_assay(self, assay_mode='lsa', assay_column_list='columns_short') -> None:
        """ loads assay """
        if assay_mode =='':
            print('no assay features')
            return None
        
        #? Yu: if the below assay modes are not used, remove them.
        if assay_mode == 'biobert-last':
            with open(self.root/('assay_features_dmis-lab_biobert-large-cased-v1.1_last_layer.npy'), 'rb') as f:
                return np.load(f, allow_pickle=True)
        elif assay_mode == 'biobert-two-last':
            with open(self.root/('assay_features_dmis-lab_biobert-large-cased-v1.1_penultimate_and_last_layer.npy'), 'rb') as f:
                return  np.load(f, allow_pickle=True)
        
        # load the prepared assay features
        asyfn = f'encoded_assays/assay_features_{assay_mode}_{assay_column_list}'
        try: # tries to open npz file else npy
            with open(self.root/(asyfn+".npz"), 'rb') as f:
                return sparse.load_npz(f)
        except:
            with open(self.root/(asyfn+".npy"), 'rb') as f:
                return np.load(f, allow_pickle=True)
        
        return None

    def _find_splits(self) -> None:
        """
        # Yu: modify this to enable cross-validation splits.
        We assume that during the preparation of the PubChem data, compounds(assays) have been indexed 
        so that a larger compound(assay) index corresponds to a compound(assay) incorporated to PubChem later in time.
        This function finds the compound(assay) index cut-points to create three chronological disjoint splits.

        The oldest `train_size` fraction of compounds(assays) are assigned to training. 
        From the remaining compounds(assays), the oldest half are assigned to vailidation, and the newest half are assigned to test.
        Only the index cut points are stored.
        """
        if self.verbose:
            logger.info(f'Find split cut-points for compound and assay indices (train_size={self.train_size}).')

        first_cut, second_cut = self._chunk(self.num_compounds, self.train_size)
        self.compound_cut = {'train': first_cut, 'valid': second_cut}

        first_cut, second_cut = self._chunk(self.num_assays, self.train_size)
        self.assay_cut = {'train': first_cut, 'valid': second_cut}

    @staticmethod
    def _chunk(n:int, first_cut_ratio:float) -> Tuple[int, int]:
        """
        # Yu: modify this to enable cross-validation splits.
        Find the two cut points required to chunk a sequence of `n` items into three parts, 
        the first having `first_cut_ratio` of the items, 
        the second and the third having approximately the half of the remaining items.

        Params
        -------
        n: int
            Length of the sequence to chunk.
        first_cut_ratio: float
            Portion of items in the first chunk. This is the `train_size`
        
        Returns
        -------
        int, int
            Positions where the first and second cut occurs.
        """
        first_cut = int(round(first_cut_ratio * n))
        second_cut = first_cut + int(round((n - first_cut) / 2))

        return first_cut, second_cut

    def subset(
            self, 
            c_low: Optional[int] = None,
            c_high: Optional[int] = None,
            a_low: Optional[int] = None,
            a_high: Optional[int] = None,
    ) -> np.ndarray:
        """
        #Yu: will be used when the `split` is set as `time_a_c`, which requires that assay and compound ids are sorted by time.
        #Yu: Remove this function if the `split` is not set as `time_a_c` in tha later benchmarking run.
        """
        if c_low is None: # set the compound low index to 0
            c_low = 0
        if c_high is None: # set the compound high index to the number of compounds
            c_high = self.num_compounds
        if a_low is None: # set the assay low index to 0
            a_low = 0
        if a_high is None: # set the assay high index to the number of assays
            a_high = self.num_assays

        if self.verbose:
            logger.info(f'Find activity triplets where {c_low} <= compound_idx <= {c_high} and {a_low} <= assay_idx <= {a_high}.')
        
        activity_bool = np.logical_and.reduce( # take multiple Boolean conditions and combines them using logical AND across all conditions.
            (
                self.activity.row >= c_low,
                self.activity.row <= c_high,
                self.activity.col >= a_low,
                self.activity.col <= a_high
            )
        )

        # applies the logical condition to the COO matrix and returns the indices that satisfy the condition. `flatnonzero` returns the indices of the non-zero elements, where is True(1).
        return np.flatnonzero(activity_bool) # return a 1D array of indices of activity where all the four conditions are True.

    def get_unique_names(
            self, 
            activity_idx: Union[int, Iterable[int], slice]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        #Yu: This function in only used in `def test` when 'Multitask' is in hparams['model']. Remove this function if 'Multitask' is not used in the later benchmarking run.

        Get the unique compound and assay names within the `activity` triplets  indexed by `activity_idx` in default, COO style.

        Params:
        -------
        activity_idx: int, iterable of int, slice
            Index to one or multiple `activity` triplets.
        
        Returns:
        -------
        compound_names: :class:`pandas.DataFrame`
        assay_names: :class:`pandas.DataFrame`
        """

        compound_idx = self.activity.row[activity_idx]
        assay_idx = self.activity.col[activity_idx]

        if isinstance(compound_idx, np.ndarray) and isinstance(assay_idx, np.ndarray):
            compound_idx = pd.unique(compound_idx)
            assay_idx = pd.unique(assay_idx)
        
        elif isinstance(compound_idx, (int, np.integer)) and isinstance(assay_idx, (int, np.integer)):
            pass # a single index means a single compound and assay, so no need to do anything.

        else:
            raise ValueError('activity_idx must be an int, iterable of int, or slice.')

        compound_names = self.compound_names.iloc[compound_idx]
        assay_names = self.assay_names.iloc[assay_idx]

        return compound_names.sort_index(), assay_names.sort_index() # sort the names alphabetically

    def getitem(
            self,
            activity_idx: Union[int, Iterable[int], slice],
            ret_np=False
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:#Yu：check the number of return values.
        """
        Get the data for the activity triplets indexed by `activity_idx` in default, COO style.
        
        Params
        -------
        activity_idx: int, iterable of int, slice
            Specifies the indices of the activity triplets to retrieve.
        ret_np: bool
            Determines the format of the returned data. If True, returns numpy arrays; If False, returns PyTorch tensors.
        
        Returns
        -------
        tuple of :class:`torch.Tensor`
        - `activity_idx`: the original indices provided as input. This will enable to reconstruct the order in which the dataset has been visited.
        - `compound_features`: shape(len(activity_idx), compound_features_size)
        - `assay_features`: shape(len(activity_idx), assay_features_size)
        - `activity`: shape(len(activity_idx), ).
        """
        compound_idx = self.activity.row[activity_idx]
        assay_idx = self.activity.col[activity_idx]
        activity = self.activity.data[activity_idx]

        # ===== get compound_features =====
        if isinstance(self.compound_features, pd.DataFrame):
            #logger.info('Compound features are stored in a DataFrame, converting to numpy array.')
            compound_smiles = self.compound_features.iloc[compound_idx]['CanonicalSMILES'].values
            from datacat4ml.Scripts.data_prep.data_featurize.compound_featurize.encode_compound  import convert_smiles_to_fp
            if self.compound_mode == 'MxFP':
                fptype = 'maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+mhfp+rdkd'
            else:
                fptype = self.compound_mode
            # Todo: fp_size as input parameter
            fp_size = 40000 #
            compound_features = convert_smiles_to_fp(compound_smiles, fp_size=fp_size, which=fptype, radius=2, njobs=1).astype(np.float32)
        else:
            #logger.info('Compound features are stored in a sparse matrix, converting to numpy array.')
            compound_features = self.compound_features[compound_idx]
            if isinstance(compound_features, sparse.csr_matrix):
                compound_features = compound_features.toarray()
        

        # ===== get assay_features =====
        assay_features = self.assay_features[assay_idx]
        if isinstance(assay_features, sparse.csr_matrix):
            assay_features = assay_features.toarray()

        # ===== Handle single indices =====
        # If `activity_idx`is a single integer or a list with only one element, the retrieved feature vectors are reshaped into 1D arrays to maintain the consistency of the output format.
        if isinstance(activity_idx, (int, np.integer)):
            compound_features = compound_features.reshape(-1) 
            assay_features = assay_features.reshape(-1) 
            activity = [activity]
        elif isinstance(activity_idx, list):
            if len(activity_idx) == 1:
                compound_features = compound_features.reshape(-1)
                assay_features = assay_features.reshape(-1)
        activity = np.array(activity)

        # ===== Return =====
        # return the data as Numpy arrays.
        if ret_np:
            return(
                activity_idx,
                compound_features, #already float32
                assay_features if not isinstance(assay_features[0], str) else assay_features, # already float32
                (float(activity)) # torch.nn.BCEWithLogitsLoss needs this to be float.
            )

        # return the data as PyTorch tensors.
        if self.compound_mode == 'smiles':
            comp_feat = compound_features
        elif isinstance(compound_features, np.ndarray):
            #print(f"compound_features type: {type(compound_features)}")
            #print(f"compound_features dtype: {getattr(compound_features, 'dtype', 'not an ndarray')}")
            #print(f"compound_features[0]: {compound_features[0]}")
            comp_feat = torch.from_numpy(compound_features)
        elif not isinstance(compound_features[0], dgl.DGLGraph): #Yu: remove this if 'graph' is not used.
            comp_feat = dgl.batch(compound_features)
        else:
            comp_feat = compound_features

        return  (
            activity_idx, 
            comp_feat, # alread float32
            torch.from_numpy(assay_features) if not isinstance(assay_features[0], str) else assay_features, # already float32
            torch.from_numpy(activity).float() # torch.nn.BCEWithLogitsLoss needs this to be float too...
        )

    def getitem_meta_assay(
            self,
            compound_idx: Union[int, List[int], slice]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For a given compound (or list), retrieve the data in the `meta-assay` style, 
        which involves summarizing assay feature vectors (positive and negative) for each compound.

        #Yu: This function is neither used in the primary code or current workflow,thus remove it if not used in the later benchmarking run.
        
        Params
        -------
        compound_idx: int, iterable of int, slice
            Index to one or multiple compounds.
        
        Returns
        -------
        tuple of :class:`torch.Tensor`
        - `compound_features`: shape(N, compound_features_size)
        - `assay_features`: shape(N, assay_features_size)
        - `activity`: shape(N, )
        """
        
        # extract the data for the specified compounds
        activity_slice = self.activity.tocsr()[compound_idx] 

        # find non-empty rows
        # `activity_slic.indptr`: pointer to the start of each row in the sparse matrix.
        # `np.diff(activity_slice.indptr)`: measures the number of elements in each row.
        # `np.where(...!=0)`: finds rows that contain non-zero elements. (i.e., rows with at least one assay-related to the compound)`
        non_empty_row_idx = np.where(np.diff(activity_slice.indptr)!=0)[0] # [0] refer to the rows of the sparse matrix, i.e., compounds.

        # initialize containers for results
        compound_features_l = [] # list of compound features
        assay_positive_features_l, assay_negative_features_l = [], [] # averaged features of positive assays, and negative assays
        activity_l = [] # activity lables

        # process each non-empty row
        for row_idx in non_empty_row_idx:
            positive_l, negative_l  = [], []
            for col_idx, activity in get_sparse_indices_and_data(activity_slice, row_idx):
                if activity == 0:
                    negative_l.append(self.assay_features[col_idx]) 
                else:
                    positive_l.append(self.assay_features[col_idx])
            
            if len(negative_l) > 0:
                compound_features_l.append(self.compound_features[row_idx])
                negative = np.vstack(negative_l).mean(axis=0)
                assay_negative_features_l.append(negative)
                activity_l.append(0)
            
            if len(positive_l) > 0:
                compound_features_l.append(self.compound_features[row_idx])
                positive = np.vstack(positive_l).mean(axis=0)
                assay_positive_features_l.append(positive)
                activity_l.append(1)

        compound_features = sparse.vstack(compound_features_l).toarray()
        assay_features_l = np.vstack(
            assay_negative_features_l + assay_positive_features_l # '+' is used to concatenate the two lists
        )

        activity = np.array(activity_l)

        return (
            torch.from_numpy(compound_features), # already float32
            torch.from_numpy(assay_features_l), # already float32
            torch.from_numpy(activity).float() # torch.nn.BCEWithLogitsLoss needs this to be float too...
        )

    @staticmethod
    def collate(batch_as_list:list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Necessary for :meth:`getitem_meta_assay` if using a :class:`torch.utils.data.DataLoader`.
        Not necessary if using :class:`torch.utils.data.BatchSampler`.

        Params
        -------
        batch_as_list: list
            Result of :meth:`getitem_meta_assay` for a mini-batch.

        Returns
        -------
        tuple of :class:`torch.Tensor`
            Data for a mini-batch.
        """
        compound_features_t, assay_features_t, activity_t = zip(*batch_as_list)
        return(
            torch.cat(compound_features_t, dim=0),
            torch.cat(assay_features_t, dim=0),
            torch.cat(activity_t, dim=0)
        )
        

    def __getitem__(self, idx: Union[int, Iterable[int], slice]) -> Tuple:
        """
        Index or slice `activity` by `idx`. The indexing mode depends on the value of `meta_assays`. 
        If False(default), the indexing is over COO triplets.
        If True, the indexing is over unique compounds.
        """

        if self.meta_assays:
            return self.getitem_meta_assay(idx)
        else:
            return self.getitem(idx)
        
    def __len__(self) -> int:
        """
        Return the length of the dataset.

        - If `meta_assays` is False (default), length is defined as the number of `(compound, assay, activity)` COO triplets.
        - If `meta_assays` is True, length is defined as the number of the unique compounds.
        """

        if self.meta_assays:
            return self.num_compounds
        else:
            return self.activity.nnz # the number of non-zero elements in the sparse matrix
        
    def __repr__(self):
        return f'InMemoryClamp\n' \
               f'\troot="{self.root}"\n' \
               f'\tcompound_mode="{self.compound_mode}"\n' \
               f'\tassay_mode="{self.assay_mode}"\n' \
               f'\ttrain_size={self.train_size}\n' \
               f'\tactivity.shape={self.activity.shape}\n' \
               f'\tactivity.nnz={self.activity.nnz}\n' \
               f'\tmeta_assays={self.meta_assays}'