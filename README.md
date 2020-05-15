## kaggle-competition

### Introduction

This repository was created for the purpose of the Kaggle competition [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview).
It was based on the repository [dl-kaggle-dataset-analysis](https://github.com/TheoHd/dl-kaggle-dataset-analysis), which goal was to create a development environment for Kaggle competitions, based on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The repository visibility is on private, and should not be shared before the end of the competition according to the [competition rules](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/rules).

Here is a sample of the actual images in the dataset :

<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/18648/logos/header.png?t=2020-02-20-17-30-35">

<small><small>Description: A sample of the Plant Pathology 2020 dataset [here is the source](https://www.cs.toronto.edu/~kriz/cifar.html)</small></small>

### Quick Start

#### Prerequisites

First of all, you need to install [Anaconda](https://www.anaconda.com/) on your computer by following this [guide]((https://docs.conda.io/projects/conda/en/latest/user-guide/install/)).

Then, in the root of the project, you have to see the command prompt with the (base) before your usual command line prefix.
<br>e.g. `(base) C:\Users\Anon\PycharmProjects\kaggle-competition>`

After that, you need to setup your [Kaggle API Key](https://github.com/Kaggle/kaggle-api#api-credentials) on your computer.

#### Environment setup

If this is the first time running this repository, launch:

```
python -m tools.create_env kaggle-competition
conda activate kaggle-competition
```

When it's complete, you can import directly the dataset by launching these two commands :

```shell script
kaggle competitions download -c plant-pathology-2020-fgvc7 # this will be directly handled by the environment created before
python -m tools.extract_dataset_zip plant-pathology-2020-fgvc7 # moves and extract data to the data/plant-pathology-2020-fgvc7 folder
```

#### Analysis Jupyter Notebook

After that, you can directly look at the notebook, by launching

```
cd notebooks
jupyter notebook
```

Feel free to read the notebook plant_pathology.ipynb, it contains all the information and analysis steps our team followed. 

### Tools

Here is a list of commands that you can use to manipulate files, and processes around the project:

```shell script
# Tensorboard
python -m tools.tb # look for generated models

# Requirements
python -m tools.refresh_req # refresh the requirements.txt file from the current Anaconda environment's packages

# Purge checkpoints
python -m tools.purge_cp # If you want to purge the checkpoints folder, you can launch:

# Purge model
#   e.g  'python -m tools.purge_model mlp_100' => deletes all files related to mlp_100 model
python -m tools.purge_model [model_name] # If you want to purge a model , you can launch:

# Testing
#   e.g ''
python -m unittest tests\test_scenarios.py # To test if the tuner works before training, you can launch the `test_scenarios.py` file with the `unittest` package from the root of the project.

# Documentation
pydoc -b # To access the documentation of the project in your browser write from the root of the project:
```

##### Performance

It took X minutes and XX seconds to build and test current model configurations.

![title](https://i.imgur.com/vI95AgZ.png)
<small><small>Test result generated the XX, mm, YYYY </small></small>

### Team members

- [Octano](https://www.kaggle.com/octano)
- [elif cilingir](https://www.kaggle.com/elifcilingir)
- [Hakim_mmz](https://www.kaggle.com/hakimmmz)
- [Fouyher](https://www.kaggle.com/fouyher)

#### TODO

- [ ] Test kaggle API in `playground.py`
- [ ] Readme : Fill performance data
- [ ] Finish README.mùd