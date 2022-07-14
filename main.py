# Logging
import logging

# Arguments
import argparse

# Text
import nltk

# NOTE: OMW-1.4 is WordNet
nltk.download(["stopwords", "punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger"])

# Project
import arguments as nlp_arguments

# Utilities
from utils import log_utils

logger = logging.getLogger()


# MAIN PARSER FUNC
def setup_args_parser():
    parser = argparse.ArgumentParser(description="NLP Project - Adversarial examples for NLP Applications")
    parser.set_defaults(func=None)
    parser.add_argument("--quiet", action="store_true", help="Flag for ignoring debug messages.")

    subparsers = parser.add_subparsers()
    nlp_arguments.setup_train_parser(subparsers.add_parser("train"))
    nlp_arguments.setup_evaluate_parser(subparsers.add_parser("evaluate"))
    nlp_arguments.setup_inference_parser(subparsers.add_parser("inference"))
    nlp_arguments.setup_attack_parser(subparsers.add_parser("attack"))

    return parser


def main():
    parser    = setup_args_parser()
    args      = parser.parse_args()

    log_utils.setup_logging(is_quiet=args.quiet)

    if args.func is None:
        parser.error("Must specify command")

    logger.debug("Arguments: %s", args)
    args.func(args)


if __name__ == '__main__':
    main()
