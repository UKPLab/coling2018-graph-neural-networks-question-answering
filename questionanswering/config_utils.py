import sys
import logging
import yaml
import random

import numpy as np
import torch

from questionanswering.grounding import staged_generation
from wikidata import endpoint_access


def load_config(config_file_path, seed=-1, gpuid=-1):
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file.read())
    print(config)
    config_global = config.get('global', {})

    logger = logging.getLogger(__name__)
    logger.setLevel(config['logger']['level'])
    ch = logging.StreamHandler()
    ch.setLevel(config['logger']['level'])
    logger.addHandler(ch)

    if seed < 0:
        np.random.seed(config_global.get('random.seed', 1))
        random.seed(seed)
        torch.manual_seed(config_global.get('random.seed', 1))
        logger.info("Seed: {}".format(config_global.get('random.seed', 1)))
    else:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        logger.info("Seed: {}".format(seed))

    if "wikidata" in config:
        endpoint_access.set_backend(config['wikidata']['backend'])

    if torch.cuda.is_available():
        logger.info("Using your CUDA device")
        if seed < 0:
            torch.cuda.manual_seed(config_global.get('random.seed', 1))
        else:
            torch.cuda.manual_seed(seed)
        if gpuid < 0:
            torch.cuda.set_device(config_global.get('gpu.id', 0))
        else:
            torch.cuda.set_device(gpuid)
        logger.info("GPU ID: {}".format(torch.cuda.current_device()))

    staged_generation.logger = logger

    return config, logger
