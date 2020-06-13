## kaggle-competition

### Introduction

This repository was created for the purpose of the Kaggle competition [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview).
It was based on the repository [dl-kaggle-dataset-analysis](https://github.com/TheoHd/dl-kaggle-dataset-analysis), which goal was to create a development environment for Kaggle competitions, based on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The repository visibility is on private, and should not be shared before the end of the competition according to the [competition rules](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/rules).

Here is a sample of the actual images in the dataset :

![A sample of the Plant Pathology 2020 dataset](https://i.imgur.com/RN0YphD.png)
<sub><sup>Source: https://www.kaggle.com/c/plant-pathology-2020-fgvc7</sup></sub>

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
### COMMON TOOLS ###

# This python command allows you to launch tensorboard in your browser
python -m tools.tb

# This python command allows you to refresh the requirements.txt file from the current Anaconda environment's packages
python -m tools.refresh_req

# This python command allows you to evaluate models
#   e.g 'python -m tools.eval_models 100 mlp' => evaluate the 100 first mlp models
#   e.g 'python -m tools.eval_models mlp' => evaluate all mlp models
python -m tools.eval_models [number_of_models] [model_name]

# This python command allows you to purge all files related to trained models
python -m tools.purge_all

# This python command allows you to purge checkpoints' folder
python -m tools.purge_cp

# This python command allows you to purge all files related to one model
#   e.g  'python -m tools.purge_model mlp_100' => deletes all files related to mlp_100 model
python -m tools.purge_model [model_name]

# This python command allows you to extract a zip in the root folder directly in the data folder
#   e.g 'python -m tools.extract_dataset_zip plant-pathology-2020-fgvc7'
python -m tools.extract_dataset_zip [zip_name_without_extension]

# This python command allows you to create an Anaconda environment from the requirements.txt file
python -m tools.create_env [environment_name]

# This python command allows you to delete an Anaconda environment
python -m tools.remove_env [environment_name]

### TESTING ###

# This python command allows you to test if the scenarios contains valid models that can be trained
python -m unittest tests\test_scenarios.py

### DOCUMENTATION ###

# This pydoc commands allows you to access documentation from your browser
pydoc -b
```

### Team members

- [Octano](https://www.kaggle.com/octano)
- [elif cilingir](https://www.kaggle.com/elifcilingir)
- [Hakim_mmz](https://www.kaggle.com/hakimmmz)
- [Fouyher](https://www.kaggle.com/fouyher)
