# Logging
import logging

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
import nltk
import nltk.corpus

# Models
import models as nlp_models

# Utilities
from utils.log_utils import TextColors
import utils.io_utils

logger = logging.getLogger(__name__)

TC = TextColors()


_POS_MAPPING = {
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

class PunctTokenizer:
    """
    Tokenizer based on nltk.word_tokenizer.
    :Language: english
    """

    def __init__(self) -> None:
        # self.sent_tokenizer = DataManager.load("TProcess.NLTKSentTokenizer")
        self.sent_tokenizer = nltk.sent_tokenize
        self.word_tokenizer = nltk.WordPunctTokenizer().tokenize
        # self.pos_tagger = DataManager.load("TProcess.NLTKPerceptronPosTagger")
        self.pos_tagger = nltk.PerceptronTagger().tag

    def tokenize(self, x : str, pos_tagging : bool = True) -> Union[ List[str], List[Tuple[str, str]] ]:
        """
        Args:
            x: A sentence.
            pos_tagging: Whether to return Pos Tagging results.
        Returns:
            A list of tokens if **pos_tagging** is `False`
            
            A list of (token, pos) tuples if **pos_tagging** is `True`
        
        POS tag must be one of the following tags: ``["noun", "verb", "adj", "adv", "other"]``
        """
        return self.do_tokenize(x, pos_tagging)
        
    def do_tokenize(self, x, pos_tagging=True):
        sentences = self.sent_tokenizer(x)
        tokens = []
        for sent in sentences:
            tokens.extend( self.word_tokenizer(sent) )

        if not pos_tagging:
            return tokens
        ret = []
        for word, pos in self.pos_tagger(tokens):
            try:
                mapped_pos = _POS_MAPPING[pos[:2]]
            except KeyError as e:
                mapped_pos = "other"
            except Exception as e:
                print(f"Different exception raised: str(e)")
            # if pos[:2] in _POS_MAPPING:
            #     mapped_pos = _POS_MAPPING[pos[:2]]
            # else:
            #     mapped_pos = "other"
            ret.append( (word, mapped_pos) )
        return ret

def detokenize(x : Union[List[str], List[Tuple[str, str]]]) -> str:
        """
        Args:
            x: The result of :py:meth:`.Tokenizer.tokenize`, can be a list of tokens or tokens with POS tags.
        Returns:
            A sentence.
        """
        if not isinstance(x, list):
            raise TypeError("`x` must be a list of tokens")
        if len(x) == 0:
            return ""
        x = [ it[0] if isinstance(it, tuple) else it for it in x ]
        return " ".join(x)


POS_LIST = ["adv", "adj", "noun", "verb", "other"]

def prefilter(token, synonym):
    if (len(synonym.split()) > 2 or (  # the synonym produced is a phrase
            synonym == token) or (  # the pos of the token synonyms are different
            token == 'be') or (
            token == 'is') or (
            token == 'are') or (
            token == 'am')):  # token is be
        return False
    else:
        return True

class WordNetSubstitute(object):

    def __init__(self, k = None):
        """
        English word substitute based on wordnet.
        Args:
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
        
        :Data Requirements: :py:data:`.TProcess.NLTKWordNet`
        :Language: english
        
        """

        self.wn = nltk.corpus.wordnet
        self.k = k

    def substitute(self, word: str, pos: str):
        if pos == "other":
            raise Exception()
        pos_in_wordnet = _WORDNET_POS_MAPPING[pos]

        wordnet_synonyms = []
        synsets = self.wn.synsets(word, pos=pos_in_wordnet)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())
        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = wordnet_synonym.name().replace('_', ' ').split()[0]
            synonyms.append(spacy_synonym)  # original word
        token = word.replace('_', ' ').split()[0]

        sss = []
        for synonym in synonyms:
            if prefilter(token, synonym):
                sss.append(synonym)
        synonyms = sss[:]

        synonyms_1 = []
        for synonym in synonyms:
            if synonym.lower() in synonyms_1:
                continue
            synonyms_1.append(synonym.lower())

        ret = []
        for syn in synonyms_1:
            ret.append((syn, 1))
        if self.k is not None:
            ret = ret[:self.k]
        return ret
    
    def __call__(self, word : str, pos : Optional[str] = None) -> List[Tuple[str, float]]:
        """
        In WordSubstitute, we return a list of words that are semantically similar to the input word.
        
        Args:
            word: A single word.
            pos: POS tag of input word. Must be one of the following: ``["adv", "adj", "noun", "verb", "other", None]``
        
        Returns:
            A list of words and their distance to original word (distance is a number between 0 and 1, with smaller indicating more similarity)
        Raises:
            WordNotInDictionaryException: input word not in the dictionary of substitute algorithm
            UnknownPOSException: invalid pos tagging
        """
        
        if pos is None:
            ret = {}
            for sub_pos in POS_LIST:
                try:
                    for word, sim in self.substitute(word, sub_pos):
                        if word not in ret:
                            ret[word] = sim
                        else:
                            ret[word] = max(ret[word], sim)
                except Exception as e:
                    continue
            list_ret = []
            for word, sim in ret.items():
                list_ret.append((word, sim))
            if len(list_ret) == 0:
                raise Exception()
            return sorted( list_ret, key=lambda x: -x[1] )
        elif pos not in POS_LIST:
            raise Exception("Invalid `pos` %s (expect %s)" % (pos, POS_LIST) )
        return self.substitute(word, pos)

class ClassifierGoal:
    def __init__(self, target, targeted):
        self.target = target
        self.targeted = targeted
    
    @property
    def is_targeted(self):
        return self.targeted

    def check(self, adversarial_sample, prediction):
        if self.targeted:
            return prediction == self.target
        else:
            return prediction != self.target


class GeneticAttacker:

    def __init__(self, 
            pop_size        : int = 20,
            max_iters       : int = 20,
            tokenizer       = None,
            substitute      = None,
            filter_words    : List[str] = None
        ):
        """
        Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018.
        `[pdf] <https://www.aclweb.org/anthology/D18-1316.pdf>`__
        `[code] <https://github.com/nesl/nlp_adversarial_examples>`__
        
        Args:
            pop_size: Genetic algorithm popluation size. **Default:** 20
            max_iter: Maximum generations of genetic algorithm. **Default:** 20
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            filter_words: A list of words that will be preserved in the attack procesudre.
        :Classifier Capacity:
            * get_pred
            * get_prob
        
        """

        self.pop_size       = pop_size
        self.max_iters      = max_iters
        
        if tokenizer is None:
            tokenizer       = PunctTokenizer()
        self.tokenizer      = tokenizer
        
        if substitute is None:
            substitute      = WordNetSubstitute()
        self.substitute     = substitute

        if filter_words is None: 
            filter_words    = nlp_dataset.STOPWORDS
        self.filter_words   = set(filter_words)

    def attack(self, victim, x_orig, goal: ClassifierGoal):
        # x_orig = x_orig.lower()
        
        x_orig = self.tokenizer.tokenize(x_orig)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        neighbours_nums = [
            self.get_neighbour_num(word, pos) if word.lower() not in self.filter_words else 0
            for word, pos in zip(x_orig, x_pos)
        ]
        neighbours = [
            self.get_neighbours(word, pos)
            if word.lower() not in self.filter_words
            else []
            for word, pos in zip(x_orig, x_pos)
        ]

        if np.sum(neighbours_nums) == 0:
            return None
        w_select_probs = neighbours_nums / np.sum(neighbours_nums)

        pop = [  # generate population
            self.perturb(
                victim, x_orig, x_orig, neighbours, w_select_probs, goal
            )
            for _ in range(self.pop_size)
        ]
        for i in range(self.max_iters):
            pop_preds = victim.get_prob(self.make_batch(pop))

            if goal.targeted:
                top_attack = np.argmax(pop_preds[:, goal.target])
                if np.argmax(pop_preds[top_attack, :]) == goal.target:
                    return detokenize(pop[top_attack])
            else:
                top_attack = np.argmax(-pop_preds[:, goal.target])
                if np.argmax(pop_preds[top_attack, :]) != goal.target:
                    return detokenize(pop[top_attack])

            pop_scores = pop_preds[:, goal.target]
            if not goal.targeted:
                pop_scores = 1.0 - pop_scores

            if np.sum(pop_scores) == 0:
                return None
            pop_scores = pop_scores / np.sum(pop_scores)

            elite = [pop[top_attack]]
            parent_indx_1 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )
            parent_indx_2 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )
            childs = [
                self.crossover(pop[p1], pop[p2])
                for p1, p2 in zip(parent_indx_1, parent_indx_2)
            ]
            childs = [
                self.perturb(
                    victim, x_cur, x_orig, neighbours, w_select_probs, goal
                )
                for x_cur in childs
            ]
            pop = elite + childs

        return None  # Failed

    def get_neighbour_num(self, word, pos):
        try:
            return len(self.substitute(word, pos))
        except Exception as e:
            return 0

    def get_neighbours(self, word, pos):
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.substitute(word, pos),
                )
            )
        except Exception as e:
            return []

    def select_best_replacements(
        self, clsf, indx, neighbours, x_cur, x_orig, goal : ClassifierGoal
        ):
        def do_replace(word):
            ret = x_cur.copy()
            ret[indx] = word
            return ret
        new_list = []
        rep_words = []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)

        pred_scores = clsf.get_prob(self.make_batch(new_list))[:, goal.target]
        if goal.targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        if np.max(new_scores) > 0:
            return new_list[np.argmax(new_scores)]
        else:
            return x_cur

    def make_batch(self, sents):
        return [detokenize(sent) for sent in sents]

    def perturb(
        self, clsf, x_cur, x_orig, neighbours, w_select_probs, goal
        ):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        if num_mods < np.sum(
            np.sign(w_select_probs)
        ):  # exists at least one indx not modified
            while x_cur[mod_idx] != x_orig[mod_idx]:  # already modified
                mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[
                    0
                ]  # random another indx
        return self.select_best_replacements(
            clsf, mod_idx, neighbours[mod_idx], x_cur, x_orig, goal
        )

    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret

    def __call__(self, victim, input_):

        if "target" in input_:
            goal = ClassifierGoal(input_["target"], targeted=True)
        else:
            origin_x = victim.get_pred([ input_ ])[0]
            goal = ClassifierGoal( origin_x, targeted=False )
        
        adversarial_sample = self.attack(victim, input_, goal)

        if adversarial_sample is not None:
            y_adv = victim.get_pred([ adversarial_sample ])[0]
            if not goal.check( adversarial_sample, y_adv ):
                raise RuntimeError("Check attacker result failed: result ([%d] %s) expect (%s%d)" % ( y_adv, adversarial_sample, "" if goal.targeted else "not ", goal.target))
        return adversarial_sample


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
        max_iters       = attacker_config.get("pop_size", 20),
        tokenizer       = None,
        substitute      = None,
        filter_words    = None
    )

    successful_predictions    = 0
    successful_attacks        = 0

    if data_type == "user":
        num_samples = len(data)

    elif data_type == "dataset":
        data = data[0]
        if data.lower() != "imdb":
            raise ValueError("Only supporting 'imdb' HuggingFace dataset.")

        # Load data
        df = nlp_dataset.create_dataframe(dataset_name=data, dataset_split=None)

        if args.num_samples != 0:
            logger.info(f"Sampling {args.num_samples} random reviews from the dataset...")
            df = df.sample(n=args.num_samples).reset_index()

        num_samples = len(df)

    with trange(num_samples) as t:
        for i in t:
            t.set_description(str(i))

            if data_type == "user":
                sample = data[i]
            elif data_type == "dataset":
                sample            = df["review"][i]
                sample_gt         = df["label"][i]
                sample_gt_name    = nlp_dataset.IMDB_LABELS[sample_gt]

            # Get original (clean) prediction
            print(f"\n=== ORIGINAL ({i}) ===")
            sample_pred_name, sample_pred_prob = classifier.get_single_pred(sample, do_print=True)

            # Run attack
            print(f"\n=== ADVERSARIAL ({i}) ===")
            adv_sample, adv_sample_pred_name, adv_sample_pred_prob = run_single_attack(
                attacker        = attacker,
                classifier      = classifier,
                input_single    = sample,
                label_names     = nlp_dataset.IMDB_LABELS,
                do_print        = True
            )

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
    failed_attacks        = successful_predictions - successful_attacks

    orig_model_accuracy   = successful_predictions / num_samples
    atk_model_accuracy    = failed_attacks / num_samples
    attack_success_rate   = successful_attacks / successful_predictions

    print("\n==================== ATTACK RESULTS ====================")
    print(f"Num samples in data:            {num_samples}")
    print(f"Original model accuracy:        {successful_predictions}/{num_samples} ({100*orig_model_accuracy:.2f}%)")
    print(f"Attacked model accuracy:        {failed_attacks}/{num_samples} ({100*atk_model_accuracy:.2f}%)")
    print(f"Attack success rate:            {successful_attacks}/{successful_predictions} ({100*attack_success_rate:.2f}%)")
