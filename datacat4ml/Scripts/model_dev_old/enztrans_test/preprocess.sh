#!/bin/bash -l
#SBATCH --job-name="enztrans_test_preprocess"
#SBATCH --partition=epyc2 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00

conda activate enztrans_test
workdir=/storage/homefs/yc24j783/datacat4ml/datacat4ml/Scripts/model_dev/enztrans_test
opennmt_py_dir=/storage/homefs/yc24j783/OpenNMT-py

# if the output directory does exist, delete it
if [ -d ${workdir}/data/preprocessed_data ]; then
    rm -r ${workdir}/data/ki_mor_1/preprocessed_data
fi

if [ -d ${workdir}/data/model ]; then
    rm -r ${workdir}/data/ki_mor_1/model
fi

if [ -d ${workdir}/data/Tensorboard ]; then
    rm -r ${workdir}/data/ki_mor_1/Tensorboard
fi

dataset_dir=${workdir}/data/ki_mor_1

# create the output directory
mkdir ${dataset_dir}/preprocessed_data
mkdir ${dataset_dir}/model
mkdir ${dataset_dir}/Tensorboard
cd $opennmt_py_dir


python preprocess.py \
        --train_src ${dataset_dir}/train_src.txt \
        --train_tgt ${dataset_dir}/train_tgt.txt \
        --valid_src ${dataset_dir}/valid_src.txt \
        --valid_tgt ${dataset_dir}/valid_tgt.txt \
        --save_data ${dataset_dir}/preprocessed_data \
        --src_seq_length 3000 \
        --tgt_seq_length 3000 \
        --src_vocab_size 3000 \
        --tgt_vocab_size 3000 \
        --share_vocab \
        --lower

WEIGHT1=1
WEIGHT2=9
batchsize=6144
dropout=0.1
rnnsize=384
wordvecsize=384
learnrate=2
layers=4
heads=8

python train.py \
        -data ${dataset_dir}/preprocessed_data \
        -save_model ${dataset_dir}/model \
        -seed 42 \
        -save_checkpoint_steps 5000 \
        -keep_checkpoint 20 \
        -train_steps 10000 \
        -param_init 0 \
        -param_init_glorot \
        -max_generator_batches 32 \
        -batch_size ${batchsize} \
        -batch_type tokens \
        -normalization tokens \
        -max_grad_norm 0 \
        -accum_count 4 \
        -optim adam \
        -adam_beta1 0.9 \
        -adam_beta2 0.998 \
        -decay_method noam \
        -warmup_steps 8000 \
        -learning_rate ${learnrate} \
        -label_smoothing 0.0 \
        -layers 4 \
        -rnn_size ${rnnsize} \
        -word_vec_size ${wordvecsize} \
        -encoder_type transformer \
        -decoder_type transformer \
        -dropout ${dropout} \
        -position_encoding \
        -global_attention general \
        -global_attention_function softmax \
        -self_attn_type scaled-dot \
        -heads 8 \
        -transformer_ff 2048 \
        #-data_ids ENZR ST_sep_aug \
        #-data_weights WEIGHT1 WEIGHT2 \
        -valid_steps 5000 \
        -valid_batch_size 4 \
        -report_every 1000 \
        -log_file ${dataset_dir}/model/training_log.txt \
        -early_stopping 10 \
        -early_stopping_criteria accuracy \
        -world_size 1 \
        -gpu_ranks 0 \
        -tensorboard \
        -tensorboard_log_dir ${dataset_dir}/Tensorboard


conda deactivate
