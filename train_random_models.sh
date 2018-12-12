#!/usr/bin/env bash
for i in {1..30};
do
    GPUID=$1
    echo "### Training Phase ###"
    MODELPATH="$(python -m questionanswering.train_model configs/train_gnn_full.yaml $i $GPUID | tail -n1)"
    echo "### Test Phase ###"
    python -m questionanswering.evaluate_on_test $MODELPATH configs/webqsp_eval_config.yaml $i $GPUID
done