# conda env: pyg (Python 3.9.16)
import argparse
import ast

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name: #--> Yu's
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    

class ParseKwargs(argparse.Action):
    """
    It processes command-line arguments into a dictionary, which can be passed as keyword arugments to a function.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)

def parse_args(args):

    """
    Yu's: I deleted the arguments about image processing, later these arguments will be replaced by the ones about SMILES processing.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data",
    )

    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.", #--> Yu's
    )

    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.", #--> Yu's
    )

    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"], #--> Yu's: `choices=["csv", "auto"]`
        default="auto", #--> Yu's: `default="csv"`
        help="Which type of dataset to process."
    )

    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t", # this means tab-separated
        help="For csv-like datasets, which separator to use."
    )

    parser.add_argument(
        "--csv-img-key", #--> Yu's: `--csv-smi-key`
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths." #--> Yu's: smiles path
    )

    parser.add_argument(
        "--csv-caption-key", #--> Yu's: caption -- assay_description, key -- label? activity? active/inactive?
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions." #--> Yu's: 
    )

    parser.add_argument(
        "--imagenet-val", #--> Yu's: `--smi-val`
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.", #--> Yu's: `Path to smiles val set for conducting zero shot evaluation.`
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )

    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )

    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        # A cooldown period refers to a phase near the end of training where the learning rate remains constant or is reduced more gradually.
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )

    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.") # To prevents overfitting by discouraging overly large weights
    parser.add_argument("--momentum", type=float, default=None, help="Momentum (for timm optimizers).") # To smoothen gradient updates and helps escape small local minima or plateaus in the loss surface.
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for." #an initial phase during training where the learning rate gradually increases from a small value to its peak.Typical range is 1000-10000 steps.
    )
    parser.add_argument(
        "--opt", type=str, default='adamw',
        help="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}']."
    )
    
    parser.add_argument(
        "--use-bn-sync", # To improves model accuracy when training with small batch sizes per GPU by using global batch statistics instead of local ones
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler", # skips the use of any learning rate scheduler, keeping the learning rate constant throughout training
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler", # specifies the type of learning rate scheduler to use
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0, # specifies the end learning rate for the cooldown schedule
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)" # >1.0: slower decay (e.g. quadratic), <1.0: faster decay
    )

    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )

    ############### Yu's ################
    #--> Yu's parser.add_argument(
    #--> Yu's     "--model",
    #--> Yu's     type=str,
    #--> Yu's     default="RN50",
    #--> Yu's     help="Name of the vision backbone to use.",
    #--> Yu's )

    parser.add_argument(
        "--smi-model",
        type=str,
        default="SMITransformer",
        help="Name of the backbone of SMILES model to use.",
    )
    ################# Yu's ################

    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )

    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )

    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    ################# Yu's ################
    # Delete the arguments below if they are not used at the end.
    #parser.add_argument(
    #    "--remote-sync",
    #    type=str,
    #    default=None,
    #    help="Optinoally sync with a remote path specified by this arg",
    #)
    #parser.add_argument(
    #    "--remote-sync-frequency",
    #    type=int,
    #    default=300,
    #    help="How frequently to sync to a remote directly if --remote-sync is not None.",
    #)
    #parser.add_argument(
    #    "--remote-sync-protocol",
    #    choices=["s3", "fsspec"],
    #    default="s3",
    #    help="How to do the remote sync backup if --remote-sync is not None.",
    #)
    ################# Yu's ################
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )

    args = parser.parse_args(args)

    if 'timm' not in args.opt:
        # set default opt params based on model name (only if timm optimizer not used)
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)

    return args