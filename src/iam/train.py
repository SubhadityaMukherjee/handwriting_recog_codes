import argparse
import logging
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import tensorflow as tf

from .datautils import *
from .models import *
from .utilsiam import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--units", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--augment", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)

    args = parser.parse_args()

    fit_ctc_model(args)
