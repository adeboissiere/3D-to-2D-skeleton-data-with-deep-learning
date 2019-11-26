3D to 2D skeleton from depth camera with Deep Learning
==============================

This mini-project aims to estimate 2D skeleton data from 3D skeleton data. More specifically 2D skeleton data for IR frames from the kinect v2. Exact formulas exist, but we are too lazy to find them ourselves. The motivation behind this work is that the NTU RGB+D dataset provides 2D IR skeleton data, while the PKU MMD dataset does not. And we need them.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make create_environment` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
