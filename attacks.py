# Logging
import logging
from re import X

# Type-hints
from typing import List, Sequence, Tuple, Optional, Union, Dict
import utils.custom_types

# Progress bar
from tqdm.auto import tqdm, trange

# Arrays
import numpy as np

# Data
import dataset as nlp_dataset

# Text
import re
import nltk
import nltk.corpus

# Models
import models as nlp_models

# Utilities
from utils.log_utils import TextColors
import utils.io_utils

logger = logging.getLogger(__name__)

TC = TextColors()


_POS_PARTS_MAPPING = {
    "JJ"    : "adj",
    "VB"    : "verb",
    "NN"    : "noun",
    "RB"    : "adv"
}

_WORDNET_POS_MAPPING = {
    "adv"   : "r",
    "adj"   : "a",
    "verb"  : "v",
    "noun"  : "n"
}

POS_LIST = list(_POS_PARTS_MAPPING.values()) + ["other"]

class CustomPunctTokenizer:
    """
    Custom tokenizer based on nltk's WordPunctTokenizer.
    """

    def __init__(self) -> None:
        self.word_tokenizer   = nltk.WordPunctTokenizer()
        self.pos_tagger       = nltk.PerceptronTagger()

    def tokenize(self, x: str, pos_tagging: bool = True) -> Union[List[str], List[Tuple[str, str]]]:
        """
        General function (matching nltk's tokenizers).

        Args:
            x -             A given text.
            pos_tagging -   Whether to return Pos Tagging results.

        Returns:
            ret - A list of either tokens or (token, pos) tuple pairs (depends on `pos_tagging`).
                NOTE: PoS tag must be one of the following tags: ``["noun", "verb", "adj", "adv", "other"]``.
        """
        return self.do_tokenize(x, pos_tagging)
        
    def do_tokenize(self, x: str, pos_tagging: bool = True) -> Union[List[str], List[Tuple[str, str]]]:
        """ Actual tokenization function. """

        tokens = self.word_tokenizer.tokenize(x)

        # Not using tagger
        if not pos_tagging:
            return tokens

        ret = []
        for word, pos in self.pos_tagger.tag(tokens):
            try:
                mapped_pos = _POS_PARTS_MAPPING[pos[:2]]
            except KeyError as e:
                mapped_pos = "other"
            except Exception as e:
                print(f"Different exception raised: str(e)")

            ret.append( (word, mapped_pos) )

        return ret

def detokenize(x : Union[List[str], List[Tuple[str, str]]]) -> str:
        """
        Convert a list of tokens into a text.

        Args:
            x -     List of tokens (or tokens with their PoS).
        Returns:
            text -  A sentence.
        """

        if not isinstance(x, list):
            raise TypeError("'x' must be a list of tokens")

        # Extract tokens
        x = [it[0] if isinstance(it, tuple) else it for it in x]

        # Combine tokens
        text = " ".join(x)

        # Remove whitespaces within quotations
        text = re.sub("\"\s+([^\"]+)\s+\"", "\""+r"\1"+"\"", text)

        # Close parentheses
        text = text.replace(" ( ", " (").replace(" ) ", ") ").replace("( ", "(").replace(" )", ")")

        # Strip "/" & "\"
        text = text.replace(" /", "/").replace("/ ", "/")
        # text = text.replace(" \\", "\\").replace("\\ ", "\\")

        # Strip single quotations
        text = text.replace(" '", "'").replace("' ", "'")

        # # Remove whitespaces for "..."
        # text = text.replace('. . .',  '...')

        # ?
        text = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", text)
        text = re.sub(r' ([.,:;?!%]+)$', r"\1", text)

        # Not
        # text = text.replace(" n't", "n't").replace("can not", "cannot")

        return text.strip()


def filter_synonyms(token: str, synonym: str) -> bool:
    """
    Filter out synonyms if:
        - Multi-worded
        - Same as token
        - Token == 'be' (in any tense )
    """
    return not (len(synonym.split()) > 2 or (token in (synonym, "be", "is", "are", "am")))


class WordNetSubstitute(object):
    """ Word substitute based on wordnet. """

    def __init__(self, k = None):
        """
        Args:
            k - Top-k results to return. If k is `None`, all results will be returned.
        """

        self.wn   = nltk.corpus.wordnet
        self.k    = k

    def substitute(self, word: str, pos: str) -> Sequence[Tuple[str, float]]:
        """
        Returns a list of (synonym, similarity score) for a given word.
        NOTE: Similarity score is fixed at 1.
        """

        if pos == "other":
            raise Exception()

        # Map PoS to match WordNet's
        pos_in_wordnet = _WORDNET_POS_MAPPING[pos]

        # Find word's synonyms
        synsets = self.wn.synsets(word, pos=pos_in_wordnet)

        # Expand using each set's lemmas
        wordnet_synonyms = []
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        # Take the first token of a multi-word synonym/input word (separated by '_')
        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = wordnet_synonym.name().replace('_', ' ').split()[0]
            synonyms.append(spacy_synonym)  # original word
        token = word.replace('_', ' ').split()[0]

        # Filter out synonyms
        synonyms = [syn for syn in synonyms if filter_synonyms(token ,syn)]

        # Lowercase & remove duplicates
        synonyms = list(dict.fromkeys([syn.lower() for syn in synonyms]))

        # Pair synonyms with fixed similarity score
        ret = [(syn, 1) for syn in synonyms]

        # Get top-k results
        if self.k is not None:
            ret = ret[:self.k]

        return ret
    
    def __call__(self, word : str, pos : Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Find a list of semantically-similar words to the input word.
        
        Args:
            word -  A single word.
            pos -   PoS tag of input word. Must be one of the following:
                    ``["adv", "adj", "noun", "verb", "other", None]``.
        
        Returns:
                    A list of words and their similarity score with the original word.
        """

        if pos is None:
            ret = {}

            # Get similar words for each PoS in the default list
            for sub_pos in POS_LIST:
                try:
                    for word, sim in self.substitute(word, sub_pos):
                        if word not in ret:
                            ret[word] = sim
                        else:  # Take highest similarity score
                            ret[word] = max(ret[word], sim)

                except Exception as e:
                    continue

            # Convert back to [(word, pos) ...]
            list_ret = list(ret.items())

            if len(list_ret) == 0:
                raise Exception()

            # Sort in descending order (from highest to lowest similarity score)
            return sorted( list_ret, key=lambda x: x[1], reverse=True )

        elif pos not in POS_LIST:
            raise Exception("Invalid `pos` %s (expect %s)" % (pos, POS_LIST) )

        return self.substitute(word, pos)

class ClassifierGoal:
    """
    Generalized classification/attack goal.
    """
    def __init__(self, target: utils.custom_types.LabelType, targeted: bool):
        self.target   = target
        self.targeted = targeted
    
    @property
    def is_targeted(self):
        return self.targeted

    def check(self, prediction: utils.custom_types.LabelType):
        """ Check correctness of prediction. """

        if self.targeted:
            return prediction == self.target
        else:
            return prediction != self.target


class GeneticAttacker:
    """
    Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary,
    Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018.
    `[pdf] <https://www.aclweb.org/anthology/D18-1316.pdf>`
    `[code] <https://github.com/nesl/nlp_adversarial_examples>`
    """

    def __init__(self, 
            pop_size        : int = 20,
            max_gens        : int = 20,
            tokenizer       = None,
            substitute      = None,
            filter_words    : List[str] = None
        ):
        """
        Args:
            pop_size -      Adversarial examples popluation size.
            max_gens -      Maximum number of generations to evolve.
            tokenizer -     A tokenizer that will be used during the attack procedure.
            substitute -    A substitute that will be used during the attack procedure.
            filter_words -  A list of words that will not be perturbed during the attack.
        """

        self.pop_size       = pop_size
        self.max_gens       = max_gens
        
        if tokenizer is None:
            tokenizer       = CustomPunctTokenizer()
        self.tokenizer      = tokenizer
        
        if substitute is None:
            substitute      = WordNetSubstitute()
        self.substitute     = substitute

        if filter_words is None: 
            filter_words    = nlp_dataset.STOPWORDS
        self.filter_words   = set(filter_words)

    def attack(
        self,
        classifier    : Union[nlp_models.LogisticRegressionSentimentClassifier, nlp_models.LSTMSentimentClassifier],
        x             : str,
        goal          : ClassifierGoal,
        ) -> str:
        """ Create an adversarial example from a given input and a classifier's output. """

        def emphasize_perturbed_token(
            adv_sample_tokens       : Sequence[str],
            perturbed_token_idxs    : Sequence[int]
            ) -> Sequence[str]:
            """ Add ANSI color formats to perturbed tokens. """

            out_adv_sample = adv_sample_tokens.copy()
            for idx in perturbed_token_idxs:
                out_adv_sample[idx] = TC.RED + out_adv_sample[idx] + TC.RESET

            return out_adv_sample

        # # Clean text
        # x_orig = nlp_dataset.clean_text(x)
        x_orig = x
        
        # Tokenize input
        x_orig = self.tokenizer.tokenize(x_orig)

        # Split into token & its PoS (Part of Speech)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        # Get the number of synonyms for each token based on its PoS
        neighbours_nums = [
            self.get_neighbour_num(word, pos)
            if word.lower() not in self.filter_words
            else 0
            for word, pos in zip(x_orig, x_pos)
        ]

        # Attack failed if no neighbours found
        if np.sum(neighbours_nums) == 0:
            return None

        # Get the synonyms for each token based on its PoS
        neighbours = [
            self.get_neighbours(word, pos)
            if word.lower() not in self.filter_words
            else []
            for word, pos in zip(x_orig, x_pos)
        ]

        # Calculate selection probabilities for perturbations
        token_selection_probs = neighbours_nums / np.sum(neighbours_nums)

        # Generate initial population
        pop = [
            self.perturb(
                classifier              = classifier,
                x_cur                   = x_orig,
                x_orig                  = x_orig,
                neighbours              = neighbours,
                token_selection_probs   = token_selection_probs,
                goal                    = goal
            ) for _ in range(self.pop_size)
        ]
        pop_mod_idxs    = list(map(lambda x: x[1], pop))
        pop             = list(map(lambda x: x[0], pop))

        for i in range(self.max_gens):
            # Get population's prediction probabilities
            pop_preds   = classifier.get_prob(self.make_batch(pop))
            pop_scores  = pop_preds[:, goal.target]

            # Check if attack is successful for current population
            # TODO: Add color to token before return
            if goal.targeted:
                top_attack = np.argmax(pop_scores)
                if np.argmax(pop_preds[top_attack, :]) == goal.target:
                    # Get all perturbed tokens' indices
                    perturbed_token_idxs = [
                        i for i, (t1, t2) in enumerate(zip(x_orig, pop[top_attack]))
                        if t1 != t2
                    ]
                    # Emphasize the perturbed tokens by coloring them red
                    out_adv_sample = emphasize_perturbed_token(
                        adv_sample_tokens       = pop[top_attack],
                        perturbed_token_idxs    = perturbed_token_idxs
                    )
                    return detokenize(out_adv_sample)
            else:
                top_attack = np.argmax(-pop_scores)
                if np.argmax(pop_preds[top_attack, :]) != goal.target:
                    # Get all perturbed tokens' indices
                    perturbed_token_idxs = [
                        i for i, (t1, t2) in enumerate(zip(x_orig, pop[top_attack]))
                        if t1 != t2
                    ]
                    # Emphasize the perturbed tokens by coloring them red
                    out_adv_sample = emphasize_perturbed_token(
                        adv_sample_tokens       = pop[top_attack],
                        perturbed_token_idxs    = perturbed_token_idxs
                    )
                    return detokenize(out_adv_sample)

            if not goal.targeted:
                pop_scores = 1.0 - pop_scores

            # Attack failed
            if np.sum(pop_scores) == 0:
                logger.warning("Attack failed. Not a single candidate found.")
                return None

            # Normalize prediction scores
            pop_scores = pop_scores / np.sum(pop_scores)

            # Best candidate
            top_candidate_mod_idx   = [pop_mod_idxs[top_attack]]
            top_candidate           = [pop[top_attack]]

            # Sample 2 parents from the population
            parent_idx_1 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )
            parent_idx_2 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )

            # Birth new candidates
            childs = [
                self.crossover(pop[p1], pop[p2])
                for p1, p2 in zip(parent_idx_1, parent_idx_2)
            ]
            childs = [
                self.perturb(
                    classifier              = classifier,
                    x_cur                   = x_cur,
                    x_orig                  = x_orig,
                    neighbours              = neighbours,
                    token_selection_probs   = token_selection_probs,
                    goal                    = goal
                ) for x_cur in childs
            ]
            childs_mod_idxs   = list(map(lambda x: x[1], childs))
            childs            = list(map(lambda x: x[0], childs))

            # Create the next population
            pop_mod_idxs      = top_candidate_mod_idx + childs_mod_idxs
            pop               = top_candidate + childs

        # Attack failed
        logger.warning("Attack failed. Reached maximum populations.")
        return None

    def get_neighbour_num(self, word: str, pos: str) -> int:
        """ Returns number of synonyms found. """

        try:
            return len(self.substitute(word, pos))

        except Exception as e:
            return 0

    def get_neighbours(self, word: str, pos: str) -> Sequence[str]:
        """ Returns list of synonyms. """

        try:
            return list(map(lambda x: x[0], self.substitute(word, pos),))

        except Exception as e:
            return []

    def select_best_replacements(
        self,
        classifier    : Union[nlp_models.LogisticRegressionSentimentClassifier, nlp_models.LSTMSentimentClassifier],
        idx           : int,
        neighbours    : Sequence[Sequence[str]],
        x_cur         : Sequence[str],
        x_orig        : Sequence[str],
        goal          : ClassifierGoal
        ):
        """
        Runs predictions on original text with all possible replacements in specified index,
        and returns the one with the lowest (or highest for targeted attacks).
        """

        def do_replace(word):
            ret         = x_cur.copy()
            ret[idx]    = word
            return ret

        # Create lists of tokens (advesarial candidates) by replacing the word
        # at index with each neighbour
        adv_candidate_list    = []
        rep_words   = []
        for word in neighbours:
            if word != x_orig[idx]:
                adv_candidate_list.append(do_replace(word))
                rep_words.append(word)

        if len(adv_candidate_list) == 0:
            return x_cur

        # Add the current for prediction comparison
        adv_candidate_list.append(x_cur)

        # Get classifier prediction probabilties for goal for each of the candidates
        pred_scores = classifier.get_prob(self.make_batch(adv_candidate_list))[:, goal.target]

        # Calculate probability difference between current candidate and rest of the list
        if goal.targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        # Return the candidate with the lowest (or highest for targeted attacks) probability
        if np.max(new_scores) > 0:
            return adv_candidate_list[np.argmax(new_scores)]
        else:
            return x_cur

    def make_batch(self, tokens_list: Sequence[Sequence[str]]) -> Sequence[str]:
        """ Create a list of texsts from a list of tokens. """

        return [detokenize(tokens) for tokens in tokens_list]

    def perturb(
        self,
        classifier              : Union[nlp_models.LogisticRegressionSentimentClassifier, nlp_models.LSTMSentimentClassifier],
        x_cur                   : Sequence[str],
        x_orig                  : Sequence[str],
        neighbours              : Sequence[Sequence[str]],
        token_selection_probs   : np.ndarray,
        goal                    : ClassifierGoal
        ) -> Tuple[Sequence[str], int]:
        """
        Create an adversarial example using a each token's neighbours.

        Args:
            classifier -            Classifier to Attack.
            x_cur -                 Current adversarial example (best candidate).
            x_orig -                Original text to perturb.
            neighbours -            List of synonyms for each token.
            token_selection_probs - Token probabilities to select for perturbations.
            goal -                  Attack goal

        Returns:
            x_cur -             Best new candidate.
            mod_idx -           Perturbed token index.
        """

        # Calculate the number of perturbations currently made to the original tokens
        x_len       = len(x_cur)
        num_mods    = sum(1 for i in range(x_len) if x_cur[i] != x_orig[i])

        # Choose token index to perturb based on selection probabilities
        mod_idx = np.random.choice(x_len, 1, p=token_selection_probs)[0]  # ?

        # At least one token hasn't been modified
        # TODO: Replace while with efficient randomization
        #   NOTE: i.e.
        #           - Calc nonzero_probs_num,
        #           - Use mod_idx to get mod_prob
        #           - Reduce mod_prob to 0.0
        #           - Increase the rest of the non-zero probs with mod_prob evenly
        if num_mods < np.sign(token_selection_probs).sum():
            while x_cur[mod_idx] != x_orig[mod_idx]:  # Already modified
                mod_idx = np.random.choice(x_len, 1, p=token_selection_probs)[0]

        # Find the most impactful (probabilty-wise + goal-wise) neighbour and
        # get the updated tokens
        x_cur = self.select_best_replacements(
            classifier    = classifier,
            idx           = mod_idx,
            neighbours    = neighbours[mod_idx],
            x_cur         = x_cur,
            x_orig        = x_orig,
            goal          = goal
        )

        return x_cur, mod_idx

    def crossover(self, x1: Sequence[str], x2: Sequence[str]):
        """ Creates child candidate by uniformly taking tokens from 2 parent candidates. """

        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret

    def __call__(
        self,
        classifier    : Union[nlp_models.LogisticRegressionSentimentClassifier, nlp_models.LSTMSentimentClassifier],
        x             : str
        ) -> str:
        """
        Create an adversarial example from an input using a classifier's output.

        Non-targeted attack:  Fool the classifier into predicting any label other than the correct one.
        Targeted:             Fool the classifier into predicting a specific non-correct label.
        """

        # Get original prediction
        y_pred    = classifier.get_pred([x])[0]

        # Set attack goal
        goal      = ClassifierGoal(y_pred, targeted=False)
        
        # Create adversarial example
        x_adv     = self.attack(classifier, x, goal)

        if x_adv is not None:
            y_adv_pred = classifier.get_pred([ x_adv ])[0]

            if not goal.check( y_adv_pred ):
                raise RuntimeError(
                    "Check attacker result failed: result ([%d] %s) expect (%s%d)" % (
                        y_adv_pred, x_adv, "" if goal.targeted else "not ", goal.target
                    )
                )

        return x_adv


def run_single_attack(
    attacker        : GeneticAttacker,
    classifier      : Union[nlp_models.LogisticRegressionSentimentClassifier, nlp_models.LSTMSentimentClassifier],
    input_single    : str,
    label_names     : Sequence[str],
    do_print        : bool = False
    ) -> Tuple[str, str, float]:
    """
    Runs a Genetic Attack on a classifier.
    NOTE: Only supporting HuggingFace IMDB sentiment analysis dataset.
    """

    adv_sample = attacker(classifier, input_single)
    adv_sample_pred_name, adv_sample_pred_prob = None, None

    if adv_sample is not None:
        adv_sample_pred_name, adv_sample_pred_prob = classifier.get_single_pred(adv_sample)

        if do_print:
            adv_sample_top_pred   = label_names.index(adv_sample_pred_name)
            pred_color            = TC.BGREEN if adv_sample_top_pred else TC.BRED

            print(f"ADVERSARIAL: \"{adv_sample}\"")
            print(f"PREDICTION:  {pred_color}{adv_sample_pred_name}{TC.RESET} ({100*adv_sample_pred_prob:.2f}%)")

    return adv_sample, adv_sample_pred_name, adv_sample_pred_prob


def attack(args):
    """
    Main attack pipeline.
    NOTE: Currently supporting only HuggingFace's IMDB sentiment analysis dataset for type 'dataset'.
    """

    # Get arguments
    model_type                  = args.model_type.lower()
    config                      = args.config
    feature_transformer_path    = args.feature_transformer_path
    model_path                  = args.model_path
    data                        = args.data
    data_type                   = args.data_type

    # CONFIG
    config = utils.io_utils.load_config(config)
    try: 
        classifier_config   = config["classifier"][model_type]
        attacker_config     = config["attacker"]
    except KeyError as e:
        raise KeyError(f"Configuration file does not contain configuration for model type: '{model_type}'.")

    # Load classifier
    if model_type == "sklearn":
        classifier = nlp_models.LogisticRegressionSentimentClassifier(
            label_names           = nlp_dataset.IMDB_LABELS,
            feature_transformer   = feature_transformer_path,
            model                 = model_path,
            do_remove_stopwords   = classifier_config.get("preprocess", {}).get("do_remove_stopwords", False)
        )
    elif model_type == "keras":
        classifier = nlp_models.LSTMSentimentClassifier(
            label_names           = nlp_dataset.IMDB_LABELS,
            feature_transformer   = feature_transformer_path,
            model                 = model_path,
            max_seq_len           = classifier_config.get("transform", {}).get("maxlen", 250)
        )

    # Instantiate attacker
    attacker = GeneticAttacker(
        pop_size        = attacker_config.get("pop_size", 20),
        max_gens        = attacker_config.get("max_gens", 20),
        tokenizer       = None,
        substitute      = None,
        filter_words    = None
    )

    successful_predictions    = 0
    successful_attacks        = 0
    total_tokens_num          = 0
    total_perturbed_tokens    = 0

    if data_type == "user":
        num_samples = len(data)

    elif data_type == "dataset":
        data = data[0]
        if data.lower() != "imdb":
            raise ValueError("Only supporting 'imdb' HuggingFace dataset.")

        # Load data
        df = nlp_dataset.create_dataframe(dataset_name=data, dataset_split=None)

        if args.num_samples != 0:
            logger.info(f"Sampling {args.num_samples}/{len(df)} random reviews from the dataset...")
            df = df.sample(n=args.num_samples).reset_index()

        num_samples = len(df)

    with trange(num_samples) as t:
        for i in t:
            t.set_description(str(i))

            # Default ground truth value
            sample_gt_name = None

            if data_type == "user":
                sample = data[i]
            elif data_type == "dataset":
                sample            = df["review"][i]
                sample_gt         = df["label"][i]
                sample_gt_name    = nlp_dataset.IMDB_LABELS[sample_gt]

            # Get original (clean) prediction
            sample_pred_name, sample_pred_prob = classifier.get_single_pred(sample, do_print=False)

            pred_color = TC.BGREEN if nlp_dataset.IMDB_LABELS.index(sample_pred_name) else TC.BRED

            # Tokenize sample
            orig_tokens       = attacker.tokenizer.tokenize(sample, pos_tagging=False)
            orig_tokens_num   = len(orig_tokens)

            print(f"\n\n=== ORIGINAL ({i}) ===")
            print(f"REVIEW: \"{sample}\"")
            print(f"PREDICTION: {pred_color}{sample_pred_name}{TC.RESET} ({100*sample_pred_prob:.2f}%)")
            if sample_gt_name is not None:
                print(f"GROUND TRUTH: {sample_gt_name}")

            # Run attack
            adv_sample, adv_sample_pred_name, adv_sample_pred_prob = run_single_attack(
                attacker        = attacker,
                classifier      = classifier,
                input_single    = sample,
                label_names     = nlp_dataset.IMDB_LABELS,
                do_print        = False
            )

            if adv_sample is not None:
                pred_color = TC.BGREEN if nlp_dataset.IMDB_LABELS.index(adv_sample_pred_name) else TC.BRED

                # Tokenize sample
                adv_tokens        = attacker.tokenizer.tokenize(re.sub(TC.ANSI_REGEX, '', adv_sample), pos_tagging=False)
                adv_tokens_num    = len(adv_tokens)

                if orig_tokens_num != adv_tokens_num:
                    logger.warning(f"Different number of tokens between original ({orig_tokens_num}) and adversarial ({adv_tokens_num})")
                else:
                    # Update total
                    total_tokens_num  += orig_tokens_num

                    # Calculate number of perturbed tokens
                    perturbed_token_idxs = [i for i, (t1, t2) in enumerate(zip(orig_tokens, adv_tokens))if t1 != t2]
                    perturbed_tokens_num = len(perturbed_token_idxs)
                    # perturbed_token_perc = perturbed_tokens_num / orig_tokens_num

                    # Update total
                    total_perturbed_tokens += perturbed_tokens_num

                print(f"\n=== ADVERSARIAL ({i}) ===")
                print(f"ADVERSARIAL: \"{adv_sample}\"")
                print(f"PREDICTION:  {pred_color}{adv_sample_pred_name}{TC.RESET} ({100*adv_sample_pred_prob:.2f}%)/n")

            # METRICS
            # Original
            if data_type == "user":
                # NOTE: Since we have no ground truth, we assume a successful prediction via thresholding
                good_pred = 0
                if sample_pred_prob > 0.7:
                    good_pred = 1
                successful_predictions += 1

            elif data_type == "dataset":
                good_pred                 = sample_pred_name == sample_gt_name
                successful_predictions    += good_pred

            # Attack
            if adv_sample is not None:
                good_attack           = adv_sample_pred_name != sample_pred_name
                successful_attacks    += (good_attack and good_pred)

            t.set_postfix(good_preds=f"{successful_predictions}/{i+1}", good_attacks=f"{successful_attacks}/{successful_predictions}")

    # Print attack results
    failed_attacks                = successful_predictions - successful_attacks

    orig_model_accuracy           = successful_predictions / num_samples
    atk_model_accuracy            = failed_attacks / num_samples
    attack_success_rate           = successful_attacks / successful_predictions
    total_perturbed_tokens_perc   = total_perturbed_tokens / total_tokens_num

    print("\n==================== ATTACK RESULTS ====================")
    print(f"Num samples in data:            {num_samples}")
    print(f"Original model accuracy:        {successful_predictions}/{num_samples} ({100*orig_model_accuracy:.2f}%)")
    print(f"Attacked model accuracy:        {failed_attacks}/{num_samples} ({100*atk_model_accuracy:.2f}%)")
    print(f"Attack success rate:            {successful_attacks}/{successful_predictions} ({100*attack_success_rate:.2f}%)")
    print(f"Avg. Perturbed word %:          {total_perturbed_tokens}/{total_tokens_num} ({100*total_perturbed_tokens_perc:.2f}%)")
