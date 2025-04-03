# SocialMediaTrends

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project examines social media interaction patterns on various platforms to determine the factors of content virality. Examining interactions such as likes, shares, comments, and views, we aim to identify the decisive attributes that make a post more likely to be virally transmitted. Using machine learning techniques, we construct predictive models to measure new content's likely virality based on historical trends and engagement patterns unique to each platform.

## Dependencies

Ensure that you have the following dependencies installed: 
- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MkDocs

Install all dependencies using:
- pip install -r requirements.txt

## Setup Instructions

Create a virtual environemnt and activate:
- python -m venv venv  
- source venv/bin/activate  # Mac/Linux  
- venv\Scripts\activate  # Windows  

Then install required packages:
- pip install -r requirements.txt

## Running the Data Processing Pipeline

Prepare and clean the dataset by running:
- python data_processing/dataset.py
This step will handle missing values, feature extraction, and transformations.

## Training and Evaluating Models

To train the machine learning model run:
- python data_processing/modeling/train.py

To make predictions using the trained model:
- python data_processing/modeling/predict.py

To evaluate the model's performance run:
- python tests/test_data.py

This will provide key metrics such as accuracy, precision, reacall, and F-1 score.

## Reproducing Results

To ensure reproducibility, set a random seed before training:
- import numpy as np
- import random
- import torch  # if using PyTorch

- np.random.seed(42)
- random.seed(42)
- torch.manual_seed(42)

Re-run the training and evaluation scripts to reproduce the results.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         data_processing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── data_processing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes data_processing a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

