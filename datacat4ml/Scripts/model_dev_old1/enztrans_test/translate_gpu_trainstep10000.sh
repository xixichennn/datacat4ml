#!/bin/bash -l
#SBATCH --job-name="enztrans_test_translate"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00

workdir=/storage/homefs/yc24j783/datacat4ml/datacat4ml/Scripts/model_dev/enztrans_test
opennmt_py_dir=/storage/homefs/yc24j783/OpenNMT-py

conda activate enztrans_test

cd $opennmt_py_dir

dataset_dir=${workdir}/data/ki_mor_1_gpu_trainstep10000

batchsize=1024
dropout=0.1
rnnsize=384
wordvecsize=384
learnrate=2
layers=4
heads=8

python translate.py \
        -model ${dataset_dir}/model_step_10000.pt \
        -src ${dataset_dir}/test_src.txt \
        -output ${dataset_dir}/predictions.txt \
        -batch_size ${batchsize} \
        -replace_unk \
        -max_length 1000 \
        -log_probs \
        -beam_size 1 \
        -n_best 50 \

conda deactivate
