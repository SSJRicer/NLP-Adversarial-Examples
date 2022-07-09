# NLP Course Final Project Template

## Repo intro
This project aims to fool sentiment analysis classifiers using adversarial examples.

## Requirements
* Any Conda package & environment manager
    * NOTE: It might be possible to use only `pip`, but we have not tested.
* python >= 3.9
* Any jupyter package (such as `ipython`)
* pandas = 1.4.0
* nltk = 3.7
* scikit-learn = 1.0.2
* tensorflow = 2.6.0 (preferably tensorflow-gpu)
* datasets = 1.15.0 (HuggingFace datasets)

## Installation

### <ins>Local machine</ins>
First, install conda using one of the following options:
* [Anaconda](https://www.anaconda.com/) (Large)
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Small)
* [Miniforge](https://github.com/conda-forge/miniforge/releases) (Small)

Then, install dependencies using the following command from a terminal:
```
conda env create --file environment.yml
```

### <ins>Google Colab</ins>
- Change runtime type's hardware accelerator from `None` to `GPU` for better performance.
    - Do this before anything, or the project files will be lost on reload.

- Install conda by entering the following commands in a code cell:
    ```jupyter cell
    ! wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    ! bash Miniforge3-Linux-x86_64.sh -b -f -p /usr/local

    import sys
    sys.path.append("/usr/local/lib/python3.9/site-packages/")
    ```

- Update conda:
    ```jupyter cell
    ! conda update -n base conda -y
    ```

- Add the dependencies' channels:
    ```jupyter cell
    ! conda config --add channels conda-forge
    ! conda config --add channels anaconda
    ```

- Upload the project's files or clone from the [github repository](https://github.com/SSJRicer/NLP-Adversarial-Examples.git) (if you have access).

- Change directory to the project's directory & run the following command to create the environment:
    ```jupyter cell
    ! conda env create -f environment.yml
    ```

- Activate the environment and run the main script using any of the 4 commands listed below:
    ```jupyter cell
    %%bash
    source activate nlp
    conda info --envs

    cd <PROJECT_DIR>
    python main.py ...
    ```



## Usage

### <ins>Data exploration</ins>
The `data_exploration.ipynb` notebook (found in `experiments/`), is used to explore and analyze the data.

### <ins>Configuration file</ins>

Prior to the starting point, a configuration JSON file is required for each of the functions.

There's a configuration file in the project's root directory, conveniently named `config.json`.

It should contain the following structure:
```config
{
    "classifier": {
        "sklearn": {
            "dataset": {
                "name" - HuggingFace dataset name.
                "path" - Path to save/load pre-processed dataframe.
            },
            "preprocess": {
                PREPROCESSING_PARAMETERS
            },
            "features": {
                "type" - 'tfidf' or 'bow'
                FEATURES_PARAMETERS
            },
            "transform": {
                TRANSFORM_PARAMETERS (Currently has none)
            },
            "model": {
                "build": {
                    BUILD_PARAMETERS
                },
                "train": {
                    TRAIN_PARAMETERS (Currently has none)
                }
            }
        },
        "keras": {
            "dataset": {
                "name" - HuggingFace dataset name.
                "path" - Path to save/load pre-processed dataframe.
            },
            "preprocess": {
                PREPROCESSING_PARAMETERS (Currently has none)
            },
            "features": {
                "type" - 'tokenizer'
                FEATURES_PARAMETERS
            },
            "transform": {
                TRANSFORM_PARAMETERS
            },
            "model": {
                "build": {
                    BUILD_PARAMETERS
                },
                "train": {
                    TRAIN_PARAMETERS
                }
            }
        }
    }
    "attacker": {
        ATTACK_PARAMETERS
    }
}
```
* `PREPROCESSING_PARAMETERS` - The `preprocess_dataframe` function found in `dataset.py` contains information about each model type's parameter options.
* `FEATURES_PARAMETERS` - The `create_and_fit_feature_transform` function found in `features.py` contains information about each model type's parameter options.
* `TRANSFORM_PARAMETERS` - The `transform_features` function found in `features.py` contains information about each model type's parameter options.
* `BUILD_PARAMETERS` - The `build_model` function found in `models.py` contains information about each model type's parameter options.
* `TRAIN_PARAMETERS` - The `train_model` function found in `train.py` contains information about each model type's parameter options.
* `ATTACK_PARAMETERS` - The `GeneticAttacker` class' `init` function found in `attacks.py` contains information about each model type's parameter options.

### <ins>Starting point</ins>

To get information regarding the different options, run the following command:
```python
python main.py -h
```

The main script has 4 major functions (each has its own arguments):
1. Training:
    ```python
    python main.py train --model-type <MODEL_TYPE> --config <CONFIG_PATH> --feature-transformer-path <TRANSFORMER_PATH> --model-path <MODEL_PATH> [--evaluate]
    ```
    * `<MODEL_TYPE>` - Model to train ('sklearn' or 'keras').
    * `<CONFIG_PATH>` - Path to configuration file.
    * `<TRANSFORMER_PATH>` - Path to save fitted feature transformer.
    * `<MODEL_PATH>` - Path to save trained model.
        * For `sklearn` model_type, you can add a `{test_accuracy:.4f}` format to the filename for storing the resulting test set accuracy.
            * i.e. `model/sklearn_tfidf_model_testAcc-{test_accuracy:.4f}.p`
        * For `keras` model_type, you can add a `{epoch:02d}` and a `{val_accuracy:.4f}` formats.
            * i.e. `model/keras_model_epoch-{epoch:02d}_valAcc-{val_accuracy:.4f}.h5`
    * `--evaluate` - [OPTIONAL] For running evaluation on the test set after training.

2. Evaluate:
    ```python
    python main.py evaluate --model-type <MODEL_TYPE> --config <CONFIG_PATH> --feature-transformer-path <TRANSFORMER_PATH> --model-path <MODEL_PATH>
    ```
    * `<MODEL_TYPE>` - Model to evaluate ('sklearn' or 'keras').
    * `<CONFIG_PATH>` - Path to configuration file.
    * `<TRANSFORMER_PATH>` - Path to load fitted feature transformer.
    * `<MODEL_PATH>` - Path to load trained model.

3. Inference:
    ```python
    python main.py inference --model-type <MODEL_TYPE> --config <CONFIG_PATH> --feature-transformer-path <TRANSFORMER_PATH> --model-path <MODEL_PATH> --input <INPUT_TEXT>
    ```
    * `<MODEL_TYPE>` - Model to evaluate ('sklearn' or 'keras').
    * `<CONFIG_PATH>` - Path to configuration file.
    * `<TRANSFORMER_PATH>` - Path to load fitted feature transformer.
    * `<MODEL_PATH>` - Path to load trained model.
    * `<INPUT_TEXT>` - Text(s) to predict sentiment for

4. Attack:
    ```python
    python main.py attack --model-type <MODEL_TYPE> --config <CONFIG_PATH> --feature-transformer-path <TRANSFORMER_PATH> --model-path <MODEL_PATH> --data <DATA> --data-type <DATA_TYPE> --num-samples <DATASET_NUM_SAMPLES>
    ```
    * `<MODEL_TYPE>` - Model to evaluate ('sklearn' or 'keras').
    * `<CONFIG_PATH>` - Path to configuration file.
    * `<TRANSFORMER_PATH>` - Path to load fitted feature transformer.
    * `<MODEL_PATH>` - Path to load trained model.
    * `<DATA>` - User input to attack or dataset name.
    * `<DATA_TYPE>` - Type of data to attack ('user' or 'dataset').
    * `<DATASET_NUM_SAMPLES>` - Number of random samples to attack from dataset (only used with 'dataset' type).

## Resources
[1] Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani B. Srivastava, Kai-Wei Chang.
[Generating Natural Language Adversarial Examples](https://arxiv.org/pdf/1804.07998.pdf)

[2] [HuggingFace IMDB Sentiment Analysis dataset](https://huggingface.co/datasets/viewer/?dataset=imdb)