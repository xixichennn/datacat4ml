# conda env: datacat (python=3.8.2)
# for `pretrained.py`
from pathlib import Path
import json
from loguru import logger
import os
import numpy as np

import torch
import torch.nn as nn
#from datacat.utils.encode_smiles import convert_smiles_to_fp

# for `models.py`
from typing import List, Tuple

from datacat4ml.Scripts.data_prep.data_featurize.compound_featurize.encode_compound import convert_smiles_to_fp

# ================================= perceptron.py =================================
def msra_initialization(m): #?Yu does 'm' standards for matrix?
    """
    MSRA intialization of the weights of a :class:`torch.nn.Module` (a layer),
    that is, the weights of the layer are :math:`\\mathbf{W} \\sim N(0, 2 / D)`, 
    where :math:`D` is the incoming dimension. For more details, see paper:

    .. `paper`: https://arxiv.org/abs/1502.01852

    Params
    ------
    m: :class:`torch.nn.Module`
        Module (layer) whose weights should be normalized.
    """
    nn.init.normal_(m.weight, mean=0., std=np.sqrt(2. / m.in_features))
    nn.init.zeros_(m.bias)

class MultilayerPerceptron(nn.Module):
    """
    Feed-forward neural network with `feature_size` input units, `num_targets` output units, and hidden layers given by the list `hidden_layer_sizes`.
    The input layer and all hidden layers share the following generic structure

    .. math::
       
       \\text{dropout} \\Big(f \\big( \\text{norm}(W x + b) \\big) \\Big) \\text{,} #?Yu what does 'big' and 'Big' in this equation mean?
    
    where

    - :math:`x` is the input to the layer,
    - :math:`W` and :math:`b` are learnanle weights,
    - :math:`\\text{norm}` is a placeholder for a normalization layer (leave empty for no normalization),
    - :math:`f` is a placeholder for an activation function (leave empty for no non-linearity), #?Yu 1, why not 'no activation' here?, 2, what does linearity mean?
    - :math:`\\text{dropout}` is a placeholder for a dropout layer (leave empty for no dropout).

    The output layer is not followed by normalization, non-linearity (this will be included in the loss function), nor dropout.
    
    """
    def __init__(self, feature_size, hidden_layer_sizes, num_targets, 
                 dropout_input=.0, dropout_hidden=.0, nonlinearity='Identity'): #?Yu `.0`?, 'Identity'?
        super().__init__()

        # linear layers
        self.linear_input = nn.Linear(feature_size, hidden_layer_sizes[0])
        self.linear_hidden_l = nn.ModuleList(
            [nn.Linear(s, spp) for s, spp in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])]
        )
        self.linear_output = nn.Linear(hidden_layer_sizes[-1], num_targets)

        # normalization layers (placeholders)
        self.normalization_input = nn.Identity()
        self.normalization_hidden_l = nn.ModuleList(
            [nn.Identity() for _ in hidden_layer_sizes[1:]]
        )
        assert len(self.linear_hidden_l) == len(self.normalization_hidden_l), 'Something went wrong initializing the hidden layers.'

        # non-linearity and dropout (placeholders)
        self.nonlinearity = getattr(nn, nonlinearity)()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
        self.num_weight_matrices = len(hidden_layer_sizes) + 1 #?Yu why +1?

    def forward(self, x):
        x = self.linear_input(x)
        x = self.normalization_input(x)
        x = self.nonlinearity(x)
        x = self.dropout_input(x)
        if len(self.linear_hidden_l) > 0:
            for linear_hidden, normalization_hidden in zip(self.linear_hidden_l, self.normalization_hidden_l):
                x = linear_hidden(x)
                x = normalization_hidden(x)
                x = self.nonlinearity(x)
                x = self.dropout_hidden(x)
        x = self.linear_output(x)
        return x
    
    def initialize_weights(self, init):
        """
        Initialize all the weights using the method `init`.
        """
        init(self.linear_input)
        if len(self.linear_hidden_l) > 0:
            for i, _ in enumerate(self.linear_hidden_l):
                init(self.linear_hidden_l[i])
        init(self.linear_output)

class NetworkLayerNorm(MultilayerPerceptron):
    """
    Child class of :class:`MultilayerPerceptron` where

    - normalization layers are set to :class:`~torch.nn.LayerNorm`,
    - non-linearity is set to :class:`~torch.nn.__` which can be set by the argument nonlinearity,
    - dropout layers are set to :class:`~torch.nn.Dropout`,

    and the weights are initialized using :meth:`msra_initialization`.
    """

    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden, nonlinearity='ReLU'):
        super().__init__(feature_size, hidden_layer_sizes, num_targets)
        self.normalization_input = nn.LayerNorm(
            normalized_shape= self.linear_input.out_features,
            elementwise_affine=False, #?Yu 
        )
        for i, linear_hidden in enumerate(self.linear_hidden_l):
            self.normalization_hidden_l[i] = nn.LayerNorm(
                normalized_shape=linear_hidden.out_features,
                elementwise_affine=False, #?Yu
            )
        self.nonlinearity = getattr(nn, nonlinearity if nonlinearity else 'ReLU')()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
        self.initialize_weights(init=msra_initialization)
    
# ================================= models.py =================================
class DotProduct(nn.Module):
    """
    Class for :class:`DotProduct` models

    This family of models projects compound and assay feature vectors to embeddings of size `embedding_size`, 
    typically by means of separate but similar compound and assay network encoders.

    Then all the pairwise similarities between compound and assay representations are computed with their dot product.

    The default :meth:`forward` method processes compound-assay interactions in COO-like format, 
    while the :meth:`forward_dense` method does it in a matrix-factorization-like manner. #?Yu matrix-factorization-like manner?

    All subclasses of :class:`DotProduct` must implement the `_define_encoders` method, which has to return the compound and assay network encoders.
    """

    def __init__(
            self,
            compound_features_size: int,
            assay_features_size: int,
            embedding_size: int,
            **kwargs
    ) -> None:
        """
        Initialize class.

        Params
        ------
        compound_features_size: int
            Input size of the compound encoder.
        assay_features_size: int
            Input size of the assay encoder.
        embedding_size: int
            Size of the association space.
        """
        super().__init__()

        self.compound_features_size = compound_features_size
        self.assay_features_size = assay_features_size
        self.embedding_size = embedding_size
        self.norm = kwargs.get('norm', None) # l2 norm of the output #?Yu what does `norm` mean? no 'norm' in 'default.json' file.

        self.compound_encoder, self.assay_encoder = self._define_encoders(**kwargs) #?Yu: why here can be called `self._define_encoders`?
        self.hps = kwargs #?Yu can kwargs to be a variable directly?
        self._check_encoders()

    def _define_encoders(self, **kwargs):  #?Yu current function doesn't implement the defination of encoders.
        """
        All subclasses of :class:`DotProduct` must implement this method, which has to return the compound and the assay encoders.
        The encoders can be any callables yielding the compound and assay embeddings.
        Typically though, the encoders will be two instances of :class:`torch.nn.Module`, whose :meth:`torch.nn.Module.forward` methods provide the embeddings.
        """
        raise NotImplementedError(
            "All subclasses of DotProduct must implement the _define_encoders method to provide  the compound and assay encoders."
        )

    def _check_encoders(self):
        """
        Run minimal consistency checks.
        """
        assert callable(self.compound_encoder)
        assert callable(self.assay_encoder)
    
    def forward(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Take `compound_features` :math:`\\in \\mathbb{R}^{N \\times C} 
        and `assay_features` :math:\\in \\mathbb{R}^{N \\times A}`, both with :math:`N` rows.
        Project both sets of features to :math:`D` dimensions, 
        that is, `compound_embeddings` :math:`\\mathbb{R}^{N \\times D}`
        and `assay_embeddings` :math:`\\mathbb{R}^{N \\times D}`.
        Compute the row-wise dot products, thus obtaining `preactivations`
        :math:`\\in \\mathbb{R}^N`.

        Params
        ------
        compound_features: :class:`torch.Tensor`, shape (N, compound_features_size)
            Array of compound features.
        assay_features: :class:`torch.Tensor`, shape (N, assay_features_size)
            Array of assay features.
        
        Returns
        -------
        :class:`torch.Tensor`, shape (N, )
            Row-wise dot products of the compound and assay projections.
        """
        # assert compound_features.shape[0] == assay_features.shape[0] # Dimension mismatch. #?Yu not necessary?

        # Yu convert dtypes to Float
        compound_features = compound_features.float()
        assay_features = assay_features.float()   

        compound_embeddings = self.compound_encoder(compound_features) #?Yu what if the compound_features have been encoded somewhere else?
        assay_embeddings = self.assay_encoder(assay_features)

        if self.norm:
            compound_embeddings = compound_embeddings / (torch.norm(compound_embeddings, dim=1, keepdim=True) +1e-13) #?Yu 
            assay_embeddings = assay_embeddings / (torch.norm(assay_embeddings, dim=1, keepdim=True) +1e-13)

        preactivations = (compound_embeddings * assay_embeddings).sum(axis=1) # `.sum(axis=1)` sums acrosst the embedding dimension, producing shape [batch_size]

        return preactivations
    
    def forward_dense(
            self, 
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Take `compound_features` :math:`\\in \\mathbb{R}^{N \\times C}
        and `assay_features` :math:`\\in \\mathbb{R}^{M \\times A}`, where the number of rows :math:`N` and :math: `M` must not be the same.
        Project both sets of features to :math:`D` dimensions, 
        that is, `compound_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}` 
        and `assay_embeddings` :math:`\\in \\mathbb{R}^{M \\times D}`.
        Compoute all the pairwise dot products by means of a matrix multiplication, thus obtaining `preactivations` :math:`\\in \\mathbb{R}^{N \\times M}`.

        Params
        ------
        compound_features: :class:`torch.Tensor`, shape (N, compound_features_size)
            Array of compound features.
        assay_features: :class:`torch.Tensor`, shape (M, assay_features_size)
            Array of assay features.

        Returns
        -------
        :class:`torch.Tensor`, shape (N, M)
            All pairwise dot products of the compound and assay projections.
        """

        # Yu convert dtypes to Float
        compound_features = compound_features.float()
        assay_features = assay_features.float()   

        compound_embeddings = self.compound_encoder(compound_features)
        assay_embeddings = self.assay_encoder(assay_features)

        if self.norm:
            compound_embeddings = compound_embeddings / (torch.norm(compound_embeddings, dim=1, keepdim=True) +1e-13)
            assay_embeddings = assay_embeddings / (torch.norm(assay_embeddings, dim=1, keepdim=True) +1e-13)
        
        preactivations = compound_embeddings @ assay_embeddings.T
        
        return preactivations #?Yu the difference between the return of `forward` and `forward_dense`

class MLPLayerNorm(DotProduct):
    """
    Subclass of :class:`DotProduct` where compound and assay encoders are each a multilayer perceptron (MLP) with `Layer Normalization`_. #?Yu layer normalization
    
    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450 #?Yu go to have a look at this paper
    """
    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            assay_layer_sizes: List[int],
            dropout_input: float, 
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]: #?Yu callable
        """
        Define encoders as multilayer perceptrons with layer normalization.

        Params
        ------
        compound_layer_sizes: list of int
            Sizes of the hidden layers of the assay encoder.
        assay_layer_sizes: list of int
            Sizes of the hidden layers of the assay encoder.
        dropout_input: float
            Dropout rate at the input layer.
        dropout_hidden: float
            Dropout rate at the hidden layers.
        
        Returns
        -------
        tuple of callable
            - Compound encoder
            - Assay encoder
        """

        # compound mutilayer perceptron
        compound_encoder = NetworkLayerNorm( #?Yu why not `MultilayerPerceptron`?
            feature_size = self.compound_features_size, 
            hidden_layer_sizes = compound_layer_sizes, #?Yu what's the difference between `self.compound_features_size` and `compound_layer_sizes`?
            num_targets= self.embedding_size, #?Yu target
            dropout_input = dropout_input,
            dropout_hidden = dropout_hidden,
            nonlinearity = kwargs.get('nonlinearity', 'ReLU')
        )

        # assay mutilayer perceptron
        assay_encoder = NetworkLayerNorm( #?Yu why not `MultilayerPerceptron``
            feature_size = self.assay_features_size,
            hidden_layer_sizes = assay_layer_sizes,
            num_targets= self.embedding_size,
            dropout_input = dropout_input,
            dropout_hidden = dropout_hidden,
            nonlinearity = kwargs.get('nonlinearity', 'ReLU')
        )

        return compound_encoder, assay_encoder

# ================================= pretrained.py =================================
class Pretrained(DotProduct):
    CHECKPOINT_URL = None
    HP_URL = None

    def __init__(self, path_dir='./data/models/pretrained/', device='cuda:0', **kwargs): #?Yu prepare this checkpoint by myself
        self.path_dir = Path(path_dir)
        self.checkpoint = self.path_dir/"checkpoint.pt"
        self.device = device
        self.kwargs = kwargs
        self.download_weights_if_not_present()

        hp = json.load(open(self.path_dir/'hp.json', 'r'))
        self.hparams = hp

        super().__init__(**hp) #?Yu is this necessary because of this class is the subclass of DotProduct?

        cp = torch.load(self.checkpoint, map_location=device)
        self.load_state_dict(cp['model_state_dict'], strict=False)
        logger.info(f"Loaded pretrained model from {self.checkpoint}")

        # override forward function of compound encoder to enable forward with non-tensor #?Yu why?
        self.compound_encoder.old_forward = self.compound_encoder.forward
        self.compound_encoder.forward = self.compound_forward
    
    def download_weights_if_not_present(self, device='cpu'):
        """ 
        download weights if not present, , which ensures that the pretrained model files (weights and hyperparameters) are available locally.
        """
        if not os.path.exists(self.path_dir):
            if not self.CHECKPOINT_URL or not self.HP_URL:
                raise ValueError("CHECKPOINT_URL and HP_URL must be set in the derived class.")
            
            os.makedirs(self.path_dir, exist_ok=True)
            logger.info(f"Downloading checkpoint.pt from {self.CHECKPOINT_URL} to {self.path_dir}") #?Yu prepare this checkpoint by myself
            os.system(f"wegt {self.CHECKPOINT_URL} -O {self.checkpoint}")
            os.system(f"wegt {self.HP_URL} -O {Path(self.path_dir)/'hp.json'}")

    def prepro_smiles(self, smi, no_grad=True): #?Yu If compound is encoded as previously, is it okay to remove this function?
        """
        Preprocess smiles for compound encoder.
        """
        fp_size = self.compound_encoder.linear_input.weight.shape[1] #?Yu what does this mean?
        fp_input = convert_smiles_to_fp(smi, fp_size=fp_size, which=self.hparams['compound_mode'], njobs=1).astype(np.float32)
        compound_features = torch.tensor(fp_input).to(self.device)
        return compound_features

    def encode_smiles(self, smis, no_grad=True): 
        """Encode SMILES"""
        compound_features = self.prepro_smiles(smis)
        with torch.no_grad() if no_grad else torch.enable_grad():
            compound_features = self.compound_encoder(compound_features) #?Yu if compound_features already, why is it necessary to use compound_encoder again?
        return compound_features
    
    def compound_forward(self, x):
        """
        compound_encoder forward function, takes smiles or features as tensor for input
        """
        if isinstance(x[0], str):
            x = self.prepro_smiles(x)
        
        return self.compound_encoder.old_forward(x)

class PretrainedCLAMP(MLPLayerNorm, Pretrained):
    """
    This class mainly prepare the assay information.
    """
    CHECKPOINT_URL = "https://cloud.ml.jku.at/s/7nxgpAQrTr69Rp2/download/checkpoint.pt" #?Yu prepare this checkpoint by myself
    HP_URL = "https://cloud.ml.jku.at/s/dRX9TWPrF7WqnHd/download/hp.json" #?Yu prepare this hp by myself

    def __init__(self, path_dir='./data/models/clamp_clip', device='cuda:0', **kwargs): #?Yu prepare the `clamp_clip` by myself, too?
        super().__init__(path_dir, device, **kwargs)
        self.compound_featuress_size = 8192 #?Yu shall I change it later?
        self.assay_features_size = 768 #?Yu why is it 768?
        self.text_encoder = None # encoder from clip

        self.assay_encoder.old_forward = self.assay_encoder.forward  #?Yu why
        self.assay_encoder.forward = self.assay_forward #?Yu why

    def load_clip_text_encoder(self):
        import clip
        model_clip, preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_encoder = model_clip

    def prepro_text(self, txt, no_grad=True): #?Yu If the assay text is encoded previously, is it okay to remove this function?
        """Preprocess text for assay encoder"""
        import clip
        if not self.text_encoder: # if not loaded yet. self.text_encoder being None is under this condition.
            self.load_clip_text_encoder()
        tokenized_text = clip.tokenize(txt, truncate=True).to(self.device)
        assay_features = self.text_encoder.encode_text(tokenized_text).float().to(self.device)
        if no_grad:
            assay_features = assay_features.detach().requires_grad_(False)
        
        return assay_features
    
    def encode_text(self, txt, no_grad=True):
        """Encode text"""
        assay_features = self.prepro_text(txt)
        with torch.no_grad() if no_grad else torch.enable_grad():
            assay_features = self.assay_encoder(assay_features)

        return assay_features
    
    def assay_forward(self, x):
        """assay_encoder forward function, takes list of text str or features tensor as input"""
        if isinstance(x[0], str):
            x = self.prepro_text(x, no_grad=True)
        return self.assay_encoder.old_forward(x)
    
    def prepro_text(self, txt, no_grad=True):
        """preprocess text for assay encoder"""
        import clip
        if not self.text_encoder:
            self.load_clip_text_encoder()
        tokenized_text = clip.tokenize(txt, truncate=True).to(self.device) 
        assay_features = self.text_encoder.encode_text(tokenized_text).float().to(self.device)
        if no_grad:
            assay_features = assay_features.detach().requires_grad_(False)
        return assay_features

    def encode_text(self, txt, no_grad=True):
        """encode text"""
        assay_features = self.prepro_text(txt)
        with torch.no_grad() if no_grad else torch.enable_grad():
            assay_features = self.assay_encoder(assay_features)
        return assay_features