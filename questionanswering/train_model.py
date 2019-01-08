import json
import logging
import sys
import random
import datetime
import os
from typing import List

import click
import numpy as np
import torch

import fackel

from questionanswering import config_utils, _utils
from questionanswering import models
from questionanswering.models import vectorization as V
from questionanswering.models import losses


from questionanswering.construction.sentence import sentence_object_hook, Sentence


@click.command()
@click.argument('config_file_path', default="default_config.yaml")
@click.argument('seed', default=-1)
@click.argument('gpuid', default=-1)
@click.argument('model_description', default="")
@click.argument('experiment_tag', default="")
def train(config_file_path, seed, gpuid, model_description, experiment_tag):
    config, logger = config_utils.load_config(config_file_path, seed, gpuid)
    if "training" not in config:
        print("Training parameters not in the config file!")
        sys.exit()

    results_logger = None
    if 'log.results' in config['training']:
        results_logger = logging.getLogger("results_logger")
        results_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(filename=config['training']['log.results'])
        fh.setLevel(logging.INFO)
        results_logger.addHandler(fh)
        results_logger.info(str(config))

    # Load data
    if not isinstance(config['training']["path_to_dataset"], list):
        config['training']["path_to_dataset"] = [config['training']["path_to_dataset"]]
    training_dataset = []
    for path_to_train in config['training']["path_to_dataset"]:
        with open(path_to_train) as f:
            training_dataset += json.load(f,  object_hook=sentence_object_hook)
    logger.info(f"Train: {len(training_dataset)}")
    dataset_name = config['training']["path_to_dataset"][0].split("/")[-1].split(".")[0]

    if "path_to_validation" not in config['training']:
        config['training']["path_to_validation"] = config['training']["path_to_dataset"][-1]
        logger.info(f"No validation set, using part of the training data.")
    with open(config['training']["path_to_validation"]) as f:
        val_dataset = json.load(f,  object_hook=sentence_object_hook)
    logger.info(f"Validation: {len(val_dataset)}")

    wordembeddings, word2idx = V.extend_embeddings_with_special_tokens(
        *_utils.load_word_embeddings(_utils.RESOURCES_FOLDER + "../../resources/embeddings/glove/glove.6B.100d.txt")
    )
    logger.info(f"Loaded word embeddings: {wordembeddings.shape}")

    model_type = config['training']["model_type"]
    logger.info(f"Model type: {model_type}")

    V.MAX_NEGATIVE_GRAPHS = 50
    training_dataset = [s for s in training_dataset if any(scores[2] > 0.25 for g, scores in s.graphs)]
    training_samples, training_targets = pack_data(training_dataset, word2idx, model_type)
    logger.info(f"Data encoded: {[m.shape for m in training_samples]}")

    V.MAX_NEGATIVE_GRAPHS = 100
    val_dataset = [s for s in val_dataset if any(scores[2] > 0.25 for g, scores in s.graphs)]
    print(f"Val F1 upper bound: {np.average([q.graphs[0].scores[2] for q in val_dataset])}")
    val_samples, val_targets = pack_data(val_dataset, word2idx, model_type)
    logger.info(f"Val data encoded: {[m.shape for m in val_samples]}")

    encoder = models.ConvWordsEncoder(
        hp_vocab_size=wordembeddings.shape[0],
        hp_word_emb_size=wordembeddings.shape[1],
        **config['model']
    )
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = getattr(models, model_type)(encoder, **config['model'])

    def metrics(targets, predictions, validation=False):
        _, predicted_targets = torch.topk(predictions, 1, dim=-1)
        _, targets = torch.topk(targets, 1, dim=-1)
        predicted_targets = predicted_targets.squeeze(1)
        targets = targets.squeeze(1)
        cur_acc = torch.sum(predicted_targets == targets).float()
        cur_acc /= predicted_targets.size(0)
        cur_f1 = 0.0
        if validation:
            for i, q in enumerate(val_dataset):
                if i < predicted_targets.size(0):
                    idx = predicted_targets.data[i]
                    if abs(idx) < len(q.graphs):
                        cur_f1 += q.graphs[idx].scores[2]
            cur_f1 /= predicted_targets.size(0)
        return {'acc': cur_acc.data[0], 'f1': cur_f1, 'predictions': predicted_targets.data.unsqueeze(0)}

    # Save models into model specific directory
    if "save_to_dir" in config['training']:
        now = datetime.datetime.now()
        model_gated = net._gnn.hp_gated if model_type == "GNNModel" else False
        config['training']['save_to_dir'] = config['training']['save_to_dir'] + \
                                            f"{'g' if model_gated else ''}" \
                                            f"{model_type.lower()}s_{now.year}Q{now.month // 4 + 1}/"
        if not os.path.exists(config['training']['save_to_dir']):
            os.makedirs(config['training']['save_to_dir'])
    container = fackel.TorchContainer(
        torch_model=net,
        criterion=losses.VariableMarginLoss(),
        # criterion=nn.MultiMarginLoss(margin=0.5, size_average=False),
        metrics=metrics,
        optimizer_params={
            'weight_decay': 0.05,
            # 'lr': 0.01
        },
        optimizer="Adam",
        logger=logger,
        init_model_weights=True,
        description=model_description,
        **config['training']
    )

    if results_logger:
        results_logger.info("Model save to: {}".format(container._save_model_to))

    log_history = container.train(
        training_samples, training_targets,
        dev=val_samples, dev_targets=val_targets
    )

    for q in val_dataset:
        random.shuffle(q.graphs)
    if container._model_checkpoint:
        container.reload_from_saved()
    val_samples, val_targets = pack_data(val_dataset, word2idx, model_type)
    predictions = container.predict_batchwise(*val_samples)
    results = metrics(*container._torchify_data(True, val_targets), predictions, validation=True)
    _, predictions = torch.topk(predictions, 1, dim=-1)
    print(f"Acc: {results['acc']}, F1: {results['f1']}")
    print(f"Predictions head: {predictions.data[:10].view(1,-1)}")

    model_name = container._save_model_to.name
    model_gated = container._model._gnn.hp_gated if model_type == "GNNModel" else False
    # Print out the model path for the evaluation script to pick up
    if "add.results.to" in config['training']:
        print(f"Adding training results to {config['training']['add.results.to']}")
        with open(config['training']["add.results.to"], 'a+') as results_out:
            results_out.write(",".join([model_name,
                                        model_type,
                                        "Gated" if model_gated else "Simple",
                                        model_description,
                                        str(seed),
                                        dataset_name,
                                        f"{training_samples[0].shape[0]}/{len(training_dataset)}",
                                        f"{val_samples[0].shape[0]}/{len(val_dataset)}",
                                        str(len(log_history)),
                                        str(results['acc']),
                                        str(results['f1']),
                                        experiment_tag
                                        ])
                              )
            results_out.write("\n")
    # Print out the model path for the evaluation script to pick up
    print(container._save_model_to)


def pack_data(selected_questions: List[Sentence],
              word2idx,
              model_type):
    max_negative_graphs = min(max(len(s.graphs) for s in selected_questions), V.MAX_NEGATIVE_GRAPHS)
    targets = np.zeros((len(selected_questions), max_negative_graphs))
    for qi, q in enumerate(selected_questions):
        q.graphs = q.graphs[:max_negative_graphs]
        random.shuffle(q.graphs)
        for gi, g in enumerate(q.graphs):
            targets[qi, gi] = g.scores[2]

    samples = V.encode_for_model(selected_questions, model_type, word2idx)
    return samples, targets


if __name__ == "__main__":
    train()
