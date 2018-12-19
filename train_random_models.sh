#!/usr/bin/env bash
for i in {1..5};
do
    GPUID=$2
    CONFIGFILE=$1
    echo "### Training Phase ###"
    MODELPATH="$(python -m questionanswering.train_model $CONFIGFILE $i $GPUID | tail -n1)"
    echo "### Test Phase ###"
    python -m questionanswering.evaluate_on_test $MODELPATH configs/webqsp_eval_config.yaml $i $GPUID
done