"""
This file holds many variables that are used across the workflow.
"""

import os
from pathlib import Path
import sys
from enum import Enum

#====================== path ======================#
def get_project_root() -> Path:
    return Path(__file__).parent.parent

# project root directory
ROOT_PATH = get_project_root()

# subdirectories
DATACAT4ML_DIR = os.path.join(ROOT_PATH, "datacat4ml")

# Scripts directories
SCRIPTS_DIR = os.path.join(DATACAT4ML_DIR, "Scripts")


# Data directories
DATA_DIR = os.path.join(DATACAT4ML_DIR, "Data")
PREP_DATA_DIR = os.path.join(DATA_DIR, "data_prep")
# "data_fetch"
FETCH_DATA_DIR = os.path.join(DATA_DIR, "data_prep", "data_fetch")

# data_categorize
CAT_DATA_DIR = os.path.join(DATA_DIR, "data_prep", "data_categorize")
CAT_HHD_GPCR_DIR = os.path.join(CAT_DATA_DIR, "cat_hhd_gpcr")
CAT_MHD_GPCR_DIR = os.path.join(CAT_DATA_DIR, "cat_mhd_gpcr")

CAT_HHD_OR_DIR = os.path.join(CAT_DATA_DIR, "cat_hhd_or")
CAT_MHD_OR_DIR = os.path.join(CAT_DATA_DIR, "cat_mhd_or")

# data_curate
CURA_DATA_DIR = os.path.join(DATA_DIR, "data_prep", "data_curate")

CURA_HHD_GPCR_DIR = os.path.join(CURA_DATA_DIR, "cura_hhd_gpcr")
CURA_MHD_GPCR_DIR = os.path.join(CURA_DATA_DIR, "cura_mhd_gpcr")
CURA_LHD_GPCR_DIR = os.path.join(CURA_DATA_DIR, "cura_lhd_gpcr")

CURA_HHD_OR_DIR = os.path.join(CURA_DATA_DIR, "cura_hhd_or")
CURA_MHD_OR_DIR = os.path.join(CURA_DATA_DIR, "cura_mhd_or")
CURA_LHD_OR_DIR = os.path.join(CURA_DATA_DIR, "cura_lhd_or")

# data_featurize
FEAT_DATA_DIR = os.path.join(DATA_DIR, "data_prep", "data_featurize")
# To avoid on-fly featurization,  OR datasets will be featurized and saved in advance for later use.
FEAT_HHD_OR_DIR = os.path.join(FEAT_DATA_DIR, "feat_hhd_or")
FEAT_MHD_OR_DIR = os.path.join(FEAT_DATA_DIR, "feat_mhd_or")
FEAT_LHD_OR_DIR = os.path.join(FEAT_DATA_DIR, "feat_lhd_or")


# data_split
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "data_prep", "data_split")
SPLIT_HET_OR_DIR = os.path.join(SPLIT_DATA_DIR, "spl_het_or")
SPLIT_HET_GPCR_DIR = os.path.join(SPLIT_DATA_DIR, "spl_het_gpcr")
SPLIT_CAT_OR_DIR = os.path.join(SPLIT_DATA_DIR, "spl_cat_or")
SPLIT_CAT_GPCR_DIR = os.path.join(SPLIT_DATA_DIR, "spl_cat_gpcr")

# hparams
HPARAMS_DIR = os.path.join(DATA_DIR, "model_dev")

# benchmark
BMK_DIR = os.path.join(DATA_DIR, "benchmark")
BMK_CAT_DIR = os.path.join(BMK_DIR, "cat_datasets")


HYPERPARAM_SPACE_DIR = os.path.join(DATA_DIR, 'model_dev', 'hyperparam_space')
AUGMENT_SMILES = os.path.join(DATA_DIR, "data_prep", 'data_augment', 'SMILES.yml')

# Figures directories
FIG_DIR= os.path.join(DATACAT4ML_DIR, "Figures")
FETCH_FIG_DIR = os.path.join(FIG_DIR, "data_prep", "data_fetch")
FEAT_FIG_DIR = os.path.join(FIG_DIR, "data_prep", "data_featurize")
CAT_FIG_DIR = os.path.join(FIG_DIR, "data_prep", "data_categorize")
SPLIT_FIG_DIR = os.path.join(FIG_DIR, "data_prep", "data_split")

RESULT_FIG_DIR = os.path.join(FIG_DIR, "results")

# best config
BEST_CONFIG_DIR = os.path.join(DATA_DIR, 'model_dev', 'best_config')
BEST_CONFIG_CAT_DIR = os.path.join(BEST_CONFIG_DIR, 'cat_datasets')
BEST_CONFIG_HET_DIR = os.path.join(BEST_CONFIG_DIR, 'het_datasets')

# models
# inner testset
MODELS_DIR = os.path.join(DATA_DIR, 'model_dev', 'models')
MODELS_CAT_DIR = os.path.join(MODELS_DIR, 'cat_datasets')
MODELS_HET_DIR = os.path.join(MODELS_DIR, 'het_datasets')
#MODELS_INNER_DIR = os.path.join(DATA_DIR, 'model_dev', 'models', 'inner_testset')
#MODELS_INNER_MCC_DIR = os.path.join(MODELS_INNER_DIR, 'mcc')
#MODELS_INNER_ALPHA_MED_DIR = os.path.join(MODELS_INNER_DIR, 'alpha80.5')
#MODELS_INNER_ALPHA_HIGH_DIR = os.path.join(MODELS_INNER_DIR, 'alpha321.9')
## augmented data for categorized data
#MODELS_INNER_AUG_DIR = os.path.join(MODELS_INNER_DIR, 'augmented')
#MODELS_INNER_AUG_MCC_DIR = os.path.join(MODELS_INNER_AUG_DIR, 'mcc')
#MODELS_INNER_AUG_ALPHA_MED_DIR = os.path.join(MODELS_INNER_AUG_DIR, 'alpha80.5')
#MODELS_INNER_AUG_ALPHA_HIGH_DIR = os.path.join(MODELS_INNER_AUG_DIR, 'alpha321.9')
#
## assaywise testset
#MODELS_ASSAYWISE_DIR = os.path.join(DATA_DIR, 'model_dev', 'models', 'assaywise_testset')
#MODELS_ASSAYWISE_MCC_DIR = os.path.join(MODELS_ASSAYWISE_DIR, 'mcc')
#MODELS_ASSAYWISE_ALPHA_MED_DIR = os.path.join(MODELS_ASSAYWISE_DIR, 'alpha80.5')
#MODELS_ASSAYWISE_ALPHA_HIGH_DIR = os.path.join(MODELS_ASSAYWISE_DIR, 'alpha321.9')


# results
# inner testset
RESULTS_DIR = os.path.join(DATA_DIR, 'model_dev', 'results')
RESULTS_INNER_DIR = os.path.join(DATA_DIR, 'model_dev', 'results', 'inner_testset')
RESULTS_INNER_MCC_DIR = os.path.join(RESULTS_INNER_DIR, 'mcc')
RESULTS_INNER_ALPHA_MED_DIR = os.path.join(RESULTS_INNER_DIR, 'alpha80.5')
RESULTS_INNER_ALPHA_HIGH_DIR = os.path.join(RESULTS_INNER_DIR, 'alpha321.9')
# augmented data for categorized data
RESULTS_INNER_AUG_DIR = os.path.join(RESULTS_INNER_DIR, 'augmented')
RESULTS_INNER_AUG_MCC_DIR = os.path.join(RESULTS_INNER_AUG_DIR, 'mcc')
RESULTS_INNER_AUG_ALPHA_MED_DIR = os.path.join(RESULTS_INNER_AUG_DIR, 'alpha80.5')
RESULTS_INNER_AUG_ALPHA_HIGH_DIR = os.path.join(RESULTS_INNER_AUG_DIR, 'alpha321.9')

# assaywise testset
RESULTS_ASSAYWISE_DIR = os.path.join(DATA_DIR, 'model_dev', 'results', 'assaywise_testset')
RESULTS_ASSAYWISE_MCC_DIR = os.path.join(RESULTS_ASSAYWISE_DIR, 'mcc')
RESULTS_ASSAYWISE_ALPHA_MED_DIR = os.path.join(RESULTS_ASSAYWISE_DIR, 'alpha80.5')
RESULTS_ASSAYWISE_ALPHA_HIGH_DIR = os.path.join(RESULTS_ASSAYWISE_DIR, 'alpha321.9')



#======================data categorize ======================#
# data categorize
OR_uniprot_ids = ['P35372', 'P41145', 'P41143', 'P41146']
OR_chemblids = ['CHEMBL233', 'CHEMBL237', 'CHEMBL236', 'CHEMBL2014']
OR_names = ['mor', 'kor', 'dor', 'nor']
OR_name_chemblids = {'mor': 'CHEMBL233', 'kor': 'CHEMBL237', 'dor': 'CHEMBL236', 'nor': 'CHEMBL2014'}


Effects=['bind', 'agon', 'antag', 'None'] # bind: binding affinity, agon: agonism, antag: antagonism
Assays = ['RBA', 
          'G-GTP', 'G-cAMP', 'G-Ca', # G: G protein activation
          'B-arrest',
          'None'] # B: beta arrestin recruitment
Std_types=['Ki', 'IC50', 'EC50']


Tasks = ['cls', 'reg']
#File_paths = [CAT_OR_DIR, FETCH_DATA_DIR]
Confidence_scores = [8, 9]
Thr_classes=[6, 7]
Use_clusterings = [True, False]
Use_smotes = [True, False]

EFFECT_TYPE_LOOKUP = {
      'CHEMBL233': {
            'bind': {
                'RBA': {
                        'Ki': {
                              'plus':['CHEMBL1101954', 'CHEMBL753397', 'CHEMBL3888044', 'CHEMBL857863', 'CHEMBL2432432', 
                                    'CHEMBL4880042', 'CHEMBL961055', 'CHEMBL753399', 'CHEMBL3706102', 'CHEMBL4409279', 
                                    'CHEMBL2444609', 'CHEMBL753395', 'CHEMBL2427304', 'CHEMBL4687692', 'CHEMBL756363', 
                                    'CHEMBL1248546', 'CHEMBL4818339', 'CHEMBL3102176', 'CHEMBL3101317', 'CHEMBL947370', 
                                    'CHEMBL5213961', 'CHEMBL5213347', 'CHEMBL5213386', 'CHEMBL5214611'],
                              'exclude': []
                              },
                        'IC50': {
                              'plus':['CHEMBL865906','CHEMBL756578', 'CHEMBL2444609', 'CHEMBL756579', 'CHEMBL1043536',
                                      'CHEMBL5058641', 'CHEMBL4337221'],
                              'exclude':[]
                              } 
                        }
                  },
            
            'agon':{
                  'G-GTP':{
                      'EC50': {
                              'plus': ['CHEMBL4675312','CHEMBL749730','CHEMBL3271443', 'CHEMBL747703', 'CHEMBL749648', 
                                       'CHEMBL749735', 'CHEMBL3381496','CHEMBL838068','CHEMBL836650', 'CHEMBL870418'],
                              'exclude': ['CHEMBL896951']
                              }
                        },
                  'G-Ca':{
                        'EC50': {
                              'plus': ['CHEMBL4390961'],
                              'exclude': []
                              }     
                        },
                  'G-cAMP':{
                        'IC50': {
                              'plus': ['CHEMBL865906'],
                              'exclude': [] 
                              },
                        'EC50': {
                              'plus': ['CHEMBL948304','CHEMBL963246','CHEMBL4057025'],
                              'exclude': ['CHEMBL4409279']
                              }
                        },
                  'B-arrest':{
                        'EC50': {
                              'plus': ['CHEMBL2114792', 'CHEMBL1613888', 'CHEMBL1737996', 'CHEMBL1738329', 'CHEMBL1738393'],
                              'exclude': []
                              }
                              }
                  },
                  
            'antag':{
                  'G-GTP':{
                        'IC50': {
                              'plus':[],
                              'exclude':['CHEMBL907187', 'CHEMBL1019888', 'CHEMBL863310', 'CHEMBL919398', 'CHEMBL2353037', 
                                         'CHEMBL953424', 'CHEMBL1028123']
                              },
                        'Ki': {
                              'plus':[],
                              'exclude':['CHEMBL1820307', 'CHEMBL870437']
                              },
                        'Ke': {
                              'plus':['CHEMBL754574', 'CHEMBL754576', 'CHEMBL754575', 'CHEMBL2384391'],
                              'exclude':[]
                              },
                        'Kb': {
                              'plus':[],
                              'exclude':['CHEMBL5127598']
                              }
                        },
                  'B-arrest':{
                        'IC50': {
                              'plus':['CHEMBL1738030', 'CHEMBL1738425', 'CHEMBL1737856', 'CHEMBL3215256', 'CHEMBL3215164'],
                              'exclude':[]
                              }
                        }
                  }

            },

      'CHEMBL237': {
            'bind': {
                'RBA':{
                        'Ki': {
                              'plus': ['CHEMBL1101955', 'CHEMBL4880044', 'CHEMBL2432433', 'CHEMBL3888045', 'CHEMBL747228', 
                                    'CHEMBL858426', 'CHEMBL751206', 'CHEMBL4369940', 'CHEMBL4396289', 'CHEMBL2427306', 
                                    'CHEMBL4687693', 'CHEMBL751204', 'CHEMBL5213491', 'CHEMBL5213407', 'CHEMBL5213960',
                                    'CHEMBL5213346', 'CHEMBL5214598', 'CHEMBL5214693', 'CHEMBL5213452', 'CHEMBL5213929', 
                                    'CHEMBL5214243', 'CHEMBL5214360', 'CHEMBL1737954', 'CHEMBL5214333', 'CHEMBL5045609',
                                    'CHEMBL2399808', 'CHEMBL5213385'],
                              'exclude': ['CHEMBL3707541', 'CHEMBL3707516']
                              },
                        'IC50': {
                              'plus':['CHEMBL4829174', 'CHEMBL753328', 'CHEMBL5058731', 'CHEMBL5214136'],
                              'exclude':[]
                              } 
                        }
                  },
            
            'agon':{
                  'G-GTP':{
                      'EC50': {
                              'plus': ['CHEMBL4675316', 'CHEMBL755511', 'CHEMBL890208', 'CHEMBL754922','CHEMBL3271447',
                                       'CHEMBL884407', 'CHEMBL699591', 'CHEMBL754911', 'CHEMBL3888041', 'CHEMBL4409302', 
                                       'CHEMBL5127629', 'CHEMBL754923', 'CHEMBL5127647'],
                              'exclude': []
                              }
                        },
                  'G-Ca':{
                        'EC50': {
                              'plus': [],
                              'exclude': []
                              }     
                        },
                  'G-cAMP':{
                        'IC50': {
                              'plus': [],
                              'exclude': [] 
                              },
                        'EC50': {
                              'plus': ['CHEMBL4057026'],
                              'exclude': []
                              }
                        },
                  'B-arrest':{
                        'EC50': {
                              'plus': ['CHEMBL1738559', 'CHEMBL1614391', 'CHEMBL1738199', 'CHEMBL1738644'],
                              'exclude': []
                              }
                        }
                  },
                  
            'antag':{
                  'G-GTP':{
                        'IC50': {
                              'plus':[],
                              'exclude':['CHEMBL907191', 'CHEMBL1019891', 'CHEMBL953425']
                              },
                        'Ki': {
                              'plus':[],
                              'exclude':['CHEMBL887536', 'CHEMBL4036863', 'CHEMBL1820306']
                              },
                        'Ke': {
                              'plus':['CHEMBL752472', 'CHEMBL699592', 'CHEMBL2384390'],
                              'exclude':[]
                              },
                        'Kb': {
                              'plus':[],
                              'exclude':['CHEMBL5127609', 'CHEMBL5127616']
                              }
                        },
                  'B-arrest':{
                        'IC50': {
                              'plus':['CHEMBL1614509', 'CHEMBL1738366', 'CHEMBL1738399', 'CHEMBL1738328', 'CHEMBL3214933', 
                                      'CHEMBL1738392', 'CHEMBL3215009'],
                              'exclude':[]
                              }
                        }
                  }
            
     
            },

      'CHEMBL236': {
            'bind': {
                'RBA':{
                        'Ki': {
                              'plus': ['CHEMBL857860', 'CHEMBL752139', 'CHEMBL2432434', 'CHEMBL3888046', 'CHEMBL4880043', 
                               'CHEMBL752140', 'CHEMBL3101318', 'CHEMBL873918', 'CHEMBL1259346', 'CHEMBL5214596',
                               'CHEMBL5214307', 'CHEMBL5214406', 'CHEMBL5213788'],
                              'exclude': ['CHEMBL752151']},
                        'IC50': {
                              'plus':['CHEMBL750411', 'CHEMBL750117', 'CHEMBL753263', 'CHEMBL4409293', 'CHEMBL1043538',
                                      'CHEMBL5058729', 'CHEMBL757183'],
                              'exclude':[]
                              } 
                        }
                  },
            
            'agon':{
                  'G-GTP':{
                      'EC50': {
                              'plus': ['CHEMBL4675313', 'CHEMBL4705242', 'CHEMBL751917','CHEMBL3271445', 'CHEMBL666296', 
                                       'CHEMBL870420', 'CHEMBL5049549'],
                              'exclude': []
                              }
                        },
                  'G-Ca':{
                        'EC50': {
                              'plus': [],
                              'exclude': []
                              }     
                        },
                  'G-cAMP':{
                        'IC50': {
                              'plus': [],
                              'exclude': [] 
                              },
                        'EC50': {
                              'plus': [],
                              'exclude': []
                              }
                        },
                  'B-arrest':{
                        'EC50': {
                              'plus': ['CHEMBL1613999', 'CHEMBL1738034', 'CHEMBL1738507', 'CHEMBL1738434'],
                              'exclude': []
                              }
                        }
                  },
                  
            'antag':{
                  'G-GTP':{
                        'IC50': {
                              'plus':['CHEMBL3590952'],
                              'exclude':['CHEMBL907189', 'CHEMBL871133', 'CHEMBL953426']
                              },
                        'Ki': {
                              'plus':[],
                              'exclude':[]
                              },
                        'Ke': {
                              'plus':['CHEMBL665482', 'CHEMBL2384389'],
                              'exclude':[]
                              },
                        'Kb': {
                              'plus':[],
                              'exclude':[]
                              }
                        },
                  'B-arrest':{
                        'IC50': {
                              'plus':['CHEMBL1613846', 'CHEMBL1738198', 'CHEMBL1738674', 'CHEMBL1738484'],
                              'exclude':[]
                              }
                        }
                  }

            },

      'CHEMBL2014': {
            'bind': {
                'RBA':{
                        'Ki': {
                              'plus':['CHEMBL750190'],
                              'exclude':[]
                              },
                        'IC50': {
                              'plus':['CHEMBL756273', 'CHEMBL831111'],
                              'exclude':[]
                              } 
                        }
                  },
            
            'agon':{
                  'G-GTP':{
                      'EC50': {
                              'plus': ['CHEMBL756270', 'CHEMBL754508'],
                              'exclude': []
                              }
                        },
                  'G-Ca':{
                        'EC50': {
                              'plus': ['CHEMBL871100'],
                              'exclude': []
                              }     
                        },
                  'G-cAMP':{
                        'IC50': {
                              'plus': [],
                              'exclude': [] 
                              },
                        'EC50': {
                              'plus': [],
                              'exclude': []
                              }
                        },
                  'B-arrest':{
                        'EC50': {
                              'plus': [],
                              'exclude': []
                              }
                        }
                  },
                  
            'antag':{
                  'G-GTP':{
                        'IC50': {
                              'plus':[],
                              'exclude':['CHEMBL948531']
                              },
                        'Ki': {
                              'plus':[],
                              'exclude':[]
                              },
                        'Ke': {
                              'plus':['CHEMBL754583', 'CHEMBL936854'],
                              'exclude':[]
                              },
                        'Kb': {
                              'plus':[],
                              'exclude':[]
                              }
                        },
                  'B-arrest':{
                        'IC50': {
                              'plus':[],
                              'exclude':[]
                              }
                        }
                  }

            }
      }

#======================data featurize ======================#

Descriptors = {
      # fingerprints
      'FP': ['ECFP4', 'ECFP6', 'MACCS', 'RDKIT_FP', 'PHARM2D', 'ERG'],
      # physicochemical properties
      'PHYSICOCHEM': ['PHYSICOCHEM'],
      # 3D descriptors
      'THREE_D': ['SHAPE3D', 'AUTOCORR3D', 'RDF', 'MORSE', 'WHIM', 'GETAWAY'],
      # ChemBERTa tokenization
#      'TOKENS': ['TOKENS'],
#      # One-hot encoding
#      'ONEHOT': ['ONEHOT'],
#      # Graph convolutional featurization
#      'GRAPH': ['GRAPH']
}

Descriptor_cats = ['FP', 'PHYSICOCHEM', 'THREE_D', 'TOKENS', 'ONEHOT', 'GRAPH']
#======================data process ======================#
RANDOM_SEED = 42

#====================== ML models ========================
