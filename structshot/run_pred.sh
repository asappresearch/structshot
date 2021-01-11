#!/usr/bin/env bash

export BERT_MODEL=bert-base-cased
export CHECKPOINT=$1
export ALGO=$2

export OUTPUT_DIR_NAME=output-pred-model
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

python3 run_pl_pred.py --data_dir ../data/ \
--labels ../data/labels.txt \
--target_labels ../data/labels-wnut.txt \
--train_fname train \
--sup_fname support-wnut-5shot/0 \
--test_fname test-wnut \
--model_name_or_path $BERT_MODEL \
--checkpoint $CHECKPOINT \
--output_dir $OUTPUT_DIR \
--algorithm $ALGO \
--tau 0.05 \
--gpus 1
