# Logging
import logging

# Arguments
import argparse

# Type-hints
from utils import custom_types

# Project
import train as nlp_train
import evaluate as nlp_evaluate
import inference as nlp_inference
import attacks as nlp_attacks


logger = logging.getLogger(__name__)

# ========================================================================================= #
#                                      ARGUMENT PARSERS                                     #
# ========================================================================================= #

# TRAIN
def setup_train_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model-type", "-mt", choices=custom_types.ModelType.__args__, default="sklearn", help="Type of model to train.")
    parser.add_argument("--config", "-c", required=True, type=str, help="Path to confiugration file.")
    parser.add_argument("--feature-transformer-path", "-tp", required=True, help="Path to save fitted feature transformer.")
    parser.add_argument("--model-path", "-mp", required=True, help="Path to save trained model.")
    # parser.add_argument("--classifier-path", "-cp", required=True, help="Path to save trained classifier (containing model + transformer).")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test data after training.")
    parser.set_defaults(func=nlp_train.train)

# EVALUATE
def setup_evaluate_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model-type", "-mt", choices=custom_types.ModelType.__args__, default="sklearn", help="Type of model to train.")
    parser.add_argument("--config", required=True, type=str, help="Path to confiugration file.")
    parser.add_argument("--feature-transformer-path", "-tp", required=True, help="Path to fitted feature transformer.")
    parser.add_argument("--model-path", "-mp", required=True, type=str, help="Path to trained model.")
    parser.set_defaults(func=nlp_evaluate.evaluate)

# INFERENCE
def setup_inference_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model-type", "-mt", choices=custom_types.ModelType.__args__, default="sklearn", help="Type of model to train.")
    parser.add_argument("--config", required=True, type=str, help="Path to confiugration file.")
    parser.add_argument("--feature-transformer-path", "-tp", required=True, help="Path to fitted feature transformer.")
    parser.add_argument("--model-path", "-mp", required=True, type=str, help="Path to trained model.")
    parser.add_argument("--input", "-i", required=True, type=str, nargs='+', help="Input(s) for prediction.")
    parser.set_defaults(func=nlp_inference.inference)

# ATTACK
def setup_attack_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model-type", "-mt", choices=custom_types.ModelType.__args__, default="sklearn", help="Type of model to attack.")
    parser.add_argument("--config", required=True, type=str, help="Path to confiugration file.")
    parser.add_argument("--feature-transformer-path", "-tp", required=True, help="Path to fitted feature transformer.")
    parser.add_argument("--model-path", "-mp", required=True, type=str, help="Path to trained model.")
    parser.add_argument("--data", "-d", required=True, type=str, nargs='+', help="Data to attack.")
    parser.add_argument("--data-type", "-dt", required=True, choices=("user", "dataset"), type=str, help="Type of data to attack (user input or dataset name).")
    parser.add_argument("--num-samples", "-ns", type=int, default=10, help="Number of random samples to take from dataset.")
    parser.set_defaults(func=nlp_attacks.attack)
